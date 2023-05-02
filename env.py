import gym
import torch
import torch.nn as nn
import numpy as np
import random


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    """
    Actors:
    input = states (24)
    output = action probabilities (4)
    """
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out


class Critic(nn.Module):
    """
    Actors:
    input = states (24)
    output = state action scalar
    """
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(torch.cat([out, a], 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out


class DDPG(nn.Module):
    def __init__(self, states, actions, minibatch, gamma):
        super().__init__()
        self.states = states
        self.actions = actions
        self.R = []
        self.minibatch = minibatch
        self.gamma = gamma
        self.optimizer = torch.optim.Adam()

        self.actor = Actor(states, actions)
        self.critic = Critic(states, actions)
        self.target = Critic(states, actions)

        # Losses
        self.critic_loss = nn.MSELoss()

    def step(self, obsevation):
        pass
        # return action

    def update(self, state, action, reward, new_state):
        self.R += (state, action, reward, new_state)
        samples = random.sample(self.R, k=self.minibatch)

        # Target
        y_target = []
        for (state, action, reward, new_state) in samples:
            y_target.append(reward + self.gamma*self.target.forward(new_state))

        # Critic
        y_critic = []
        for (state, action, reward, new_state) in samples:
            y_critic.append(reward + self.gamma*self.target.forward(state))

        self.optimizer.zero_grad()
        loss = self.critic_loss(y_target, y_critic)
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human")
    ddpg = DDPG(states=24, actions=4)

    env.action_space.seed(42)
    state, info = env.reset(seed=42)
    for _ in range(1000):
        action = ddpg.step(state)
        new_state, reward, terminated, truncated, info = env.step(action)
        ddpg.update(state, action, reward, new_state)

        state = new_state
        if terminated or truncated:
            observation, info = env.reset()

