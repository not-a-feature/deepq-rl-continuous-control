"""
Author: Jules Kreuer / @not_a_feature
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from itertools import zip_longest

from TestBench import TestBench


# Define Actor and Critic networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Ornstein-Uhlenbeck noise
class OU_noise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


# Replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, transition):
        """
        Adds a new transition to the buffer.
        Pops the oldest transition if the buffer_size is reached.
        """
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
        else:
            self.buffer.pop(0)
            self.buffer.append(transition)

    def sample(self, batch_size):
        """Takes a random subsample of the buffer to reduce correlation."""
        res = tuple(zip(*random.sample(self.buffer, batch_size)))
        if res:
            res = (list(res[0]), res[1], list(res[2]), res[3], list(res[4]))
        return res


def sample_three_buffers(b_unknown, b_converged, b_regular, batch_size, ratio):
    """
    Samples from three buffers in a stratified manner.
    Stratification happens only in between b_converged and b_regular.
    Samples from b_unknown are samples according to the size-proportion.
    """
    total_buffer_size = len(b_unknown.buffer) + len(b_converged.buffer) + len(b_regular.buffer)
    unknown_ratio = len(b_unknown.buffer) / total_buffer_size
    n_unknown = int(batch_size * unknown_ratio)

    remaining_samples = batch_size - n_unknown

    n_converged = int(remaining_samples * ratio)
    n_regular = remaining_samples - n_converged

    if len(b_converged.buffer) < n_converged:
        n_regular += n_converged - len(b_converged.buffer)
        n_converged = len(b_converged.buffer)

    if len(b_regular.buffer) < n_regular:
        n_converged += n_regular - len(b_regular.buffer)
        n_regular = len(b_regular.buffer)

    unknown_samples = b_unknown.sample(n_unknown)
    converged_samples = b_converged.sample(n_converged)
    regular_samples = b_regular.sample(n_regular)

    def concat_group(*samples):
        return [
            sample_element for sample in samples if sample is not None for sample_element in sample
        ]

    return [
        concat_group(us, cs, rs)
        for us, cs, rs in zip_longest(
            *(unknown_samples, converged_samples, regular_samples), fillvalue=None
        )
    ]


# Update function
def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)


class DDPG(TestBench):
    def __init__(
        self,
        ENV_NAME,
        MODEL_NAME,  # Specific name of run / model
        # General Hyperparameters
        BUFFER_SIZE=5000,
        BATCH_SIZE=64,
        GAMMA=0.99,
        TAU=1e-3,
        ACTOR_LR=1e-4,
        CRITIC_LR=1e-3,
        NOISE_SCALE=0.1,
        # Stratification Parameter
        REWARD_THRESHOLD=0.5,  # stdev from mean
        THRESHOLD_WIDTH=50,  # of the last n episodes
        STRATIFY_RATIO=0.5,
        # Episode Parameters
        EPISODES=200,  # Number of episodes to run
        MAX_STEPS=200,  # Max number of steps per episode
        SAVE_EVERY=10,  # Save every n-th model
        VERBOSE=True,
    ):
        self.ENV_NAME = ENV_NAME
        self.MODEL_NAME = MODEL_NAME
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.ACTOR_LR = ACTOR_LR
        self.CRITIC_LR = CRITIC_LR
        self.NOISE_SCALE = NOISE_SCALE
        self.REWARD_THRESHOLD = REWARD_THRESHOLD
        self.THRESHOLD_WIDTH = THRESHOLD_WIDTH
        self.STRATIFY_RATIO = STRATIFY_RATIO
        self.EPISODES = EPISODES
        self.MAX_STEPS = MAX_STEPS
        self.SAVE_EVERY = SAVE_EVERY
        self.VERBOSE = VERBOSE

        self.MODEL_SAVE_DIR = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../",
            "runs",
            self.ENV_NAME,
            self.MODEL_NAME,
        )
        self.GLOBAL_PATH = os.path.join(self.MODEL_SAVE_DIR, "globals")
        self.LOSS_PATH = os.path.join(self.MODEL_SAVE_DIR, "loss.csv")

    def train(self, env):
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # Initialize networks
        actor = Actor(state_dim, action_dim).float()
        critic = Critic(state_dim, action_dim).float()
        target_actor = Actor(state_dim, action_dim).float()
        target_critic = Critic(state_dim, action_dim).float()

        # Copy the initial weights
        soft_update(target_actor, actor, 1.0)
        soft_update(target_critic, critic, 1.0)

        # Set up optimizers
        actor_optimizer = optim.Adam(actor.parameters(), lr=self.ACTOR_LR)
        critic_optimizer = optim.Adam(critic.parameters(), lr=self.CRITIC_LR)

        # Initialize the replay buffer containing episodes
        # that have not reached the end state.
        regular_buffer = ReplayBuffer(self.BUFFER_SIZE)

        # Initialize the end buffer containing episodes
        # that reached the end state.
        end_buffer = ReplayBuffer(self.BUFFER_SIZE)

        # Initialize list of last n rewards for the adaptive threshold
        last_rewards = []

        # Initialize the noise process
        noise = OU_noise(action_dim)

        for episode in range(1, self.EPISODES + 1):
            state, _ = env.reset()

            episode_reward = 0
            episode_buffer = ReplayBuffer(self.EPISODES)

            for t in range(self.MAX_STEPS):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)

                action = actor(state_tensor).detach().numpy()[0] + self.NOISE_SCALE * noise.sample()

                # Observation, Reward, Terminal state reached
                next_state, reward, termd, _, _ = env.step(action)

                # Store transition in the episode buffer
                episode_buffer.add((state, action, reward, next_state, float(termd)))

                total_buffer_size = (
                    len(episode_buffer.buffer) + len(end_buffer.buffer) + len(regular_buffer.buffer)
                )
                # Sample a random minibatch of transitions
                if total_buffer_size > self.BATCH_SIZE:
                    states, actions, rewards, next_states, termds = sample_three_buffers(
                        episode_buffer,
                        end_buffer,
                        regular_buffer,
                        self.BATCH_SIZE,
                        self.STRATIFY_RATIO,
                    )

                    states = torch.FloatTensor(np.array(states))

                    actions = torch.FloatTensor(np.array(actions))
                    rewards = torch.FloatTensor(rewards).unsqueeze(1)
                    next_states = torch.FloatTensor(np.array(next_states))
                    termds = torch.FloatTensor(termds).unsqueeze(1)

                    # Update critic
                    target_actions = target_actor(next_states)
                    y = rewards + self.GAMMA * target_critic(next_states, target_actions) * (
                        1 - termds
                    )
                    q_values = critic(states, actions)
                    critic_loss = nn.MSELoss()(q_values, y)
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()

                    # Update actor
                    actor_loss = -critic(states, actor(states)).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # Update target networks
                    soft_update(target_actor, actor, self.TAU)
                    soft_update(target_critic, critic, self.TAU)

                episode_reward += reward
                state = next_state

                if termd:
                    break
            # Adaptive episode classification

            if len(last_rewards) < self.THRESHOLD_WIDTH:
                last_rewards.append(episode_reward)
            else:
                last_rewards.pop(0)
                last_rewards.append(episode_reward)

            threshold = np.mean(last_rewards) + self.REWARD_THRESHOLD * np.std(last_rewards)

            if threshold < episode_reward:
                for t in episode_buffer.buffer:
                    end_buffer.add(t)
            else:
                for t in episode_buffer.buffer:
                    regular_buffer.add(t)

            self.log(self.LOSS_PATH, f"{episode}, {episode_reward}, {threshold}")

            if episode % self.SAVE_EVERY == 0:
                # Save model every n-th episodes
                self.save_model(actor, actor_optimizer, episode)

    def load_actor(self, env, model_path):
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        actor = Actor(state_dim, action_dim).float()
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        actor.load_state_dict(checkpoint["model"])

        return actor
