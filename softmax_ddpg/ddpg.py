"""
DDPG implementation using softmax sampling.

Author: Jonas Müller
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

from TestBench import TestBench


# Define Actor and Critic networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
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
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
            # print(transition)
        else:
            self.buffer.pop(0)
            self.buffer.append(transition)

    def weightingFunction(self, x, theta):
        x += np.abs(np.mean(x))  # center
        x = np.abs(x)
        # x = np.where(x > 0, x**2, np.abs(x)) # square positive values, while negatives remain same
        x = np.exp(np.divide(x, theta))
        x = np.divide(x, np.sum(x))
        return x

    def sample(self, batch_size, weighting=True, theta=0.2):
        if weighting:
            # buffer
            buffer = zip(*self.buffer)

            # get rewards
            states, actions, rewards, next_states, dones = buffer

            # get distribution over rewards
            rewardsValue = np.array(rewards, dtype=np.float64)
            prob = self.weightingFunction(rewardsValue, theta)

            # get indices of the highest elements
            # sample indices with prob of the weights -> keeps stochasticity in the loop
            highestIndices = np.random.choice(
                np.arange(len(rewards)), size=batch_size, replace=True, p=prob
            )

            return [
                [states[i] for i in highestIndices],
                [actions[i] for i in highestIndices],
                [rewards[i] for i in highestIndices],
                [next_states[i] for i in highestIndices],
                [dones[i] for i in highestIndices],
            ]
        else:
            return zip(*random.sample(self.buffer, batch_size))


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
        BUFFER_SIZE=1e6,
        BATCH_SIZE=64,
        GAMMA=0.99,
        TAU=1e-3,
        ACTOR_LR=1e-4,
        CRITIC_LR=1e-3,
        NOISE_SCALE=0.1,
        # Softmax Paramter
        THETA=1,
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
        self.THETA = THETA
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

    # DDPG Algorithm
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

        # Initialize the replay buffer
        replay_buffer = ReplayBuffer(self.BUFFER_SIZE)

        # Initialize the noise process
        noise = OU_noise(action_dim)

        # start saving
        rewardsOut = np.zeros(self.EPISODES)
        for episode in range(1, self.EPISODES + 1):
            state, _ = env.reset()

            episode_reward = 0

            for t in range(self.MAX_STEPS):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)

                action = actor(state_tensor).detach().numpy()[0] + self.NOISE_SCALE * noise.sample()

                next_state, reward, done, _, _ = env.step(action)

                # Store transition in the replay buffer
                replay_buffer.add((state, action, reward, next_state, float(done)))

                # Sample a random minibatch of transitions
                if len(replay_buffer.buffer) > self.BATCH_SIZE:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(
                        self.BATCH_SIZE, theta=self.THETA
                    )

                    states = torch.FloatTensor(np.array(states))

                    actions = torch.FloatTensor(np.array(actions))
                    rewards = torch.FloatTensor(rewards).unsqueeze(1)
                    next_states = torch.FloatTensor(np.array(next_states))
                    dones = torch.FloatTensor(dones).unsqueeze(1)

                    # Update critic
                    target_actions = target_actor(next_states)
                    y = rewards + self.GAMMA * target_critic(next_states, target_actions) * (
                        1 - dones
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

                if done:
                    break
            rewardsOut[episode - 1] = episode_reward
            self.log(self.LOSS_PATH, f"{episode}, {episode_reward}")

            # if episode % self.SAVE_EVERY == 0:
            # Save model every n-th episodes
            #    self.save_model(actor, actor_optimizer, episode)

        # save rewards
        np.savetxt("rewardsSoftmaxSampling.csv", rewards, delimiter=",")

    def load_actor(self, env, model_path):
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        actor = Actor(state_dim, action_dim).float()
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        actor.load_state_dict(checkpoint["model"])

        return actor
