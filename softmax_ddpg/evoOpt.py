import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
import os
from ddpg import *
import math

def ddpgRun(hyperparameters):

    # global fixed
    ENV_NAME = "Pendulum-v1"
    env = gym.make(ENV_NAME)
    episodes = 10  # 1000
    max_steps = 200  # 200
    BUFFER_SIZE = 1e6
    BATCH_SIZE = 64
    SAVE_EVERY = np.inf

    # hyperparameters variable
    GAMMA = hyperparameters["gamma"]
    TAU = hyperparameters["tau"]
    THETA = hyperparameters["theta"]
    ACTOR_LR = hyperparameters["actorLr"]
    CRITIC_LR = hyperparameters["criticLr"]
    NOISE_SCALE = hyperparameters["noise"]

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

        def softmaxTemperature(self, x, theta):
            exp_x = np.exp(x / theta)
            return exp_x / np.sum(exp_x)

        def sample(self, batch_size, weighting=False, theta = 1):
            if weighting:
                # buffer
                buffer = zip(*self.buffer)

                # get rewards
                states, actions, rewards, next_states, dones = buffer

                # get distribution over rewards
                rewardsValue = np.array(rewards, dtype=np.float64)
                prob = self.softmaxTemperature(rewardsValue, theta)

                # get indices of the highest elements
                # sample indices with prob of the weights -> keeps stochasticity in the loop
                highestIndices = np.random.choice(
                    np.arange(len(rewards)), size=batch_size, replace=True, p=prob)

                return [
                    [states[i] for i in highestIndices],
                    [actions[i] for i in highestIndices],
                    [rewards[i] for i in highestIndices],
                    [next_states[i] for i in highestIndices],
                    [dones[i] for i in highestIndices],
                ]

                #return zip(*self.buffer[highestIndices]) ## fix
            else:

                return zip(*random.sample(self.buffer, batch_size))

    def ddpg(env, episodes, max_steps):
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
        actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)
        critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)

        # Initialize the replay buffer
        replay_buffer = ReplayBuffer(BUFFER_SIZE)

        # Initialize the noise process
        noise = OU_noise(action_dim)

        # start saving
        rewardsOut = np.zeros(episodes)
        for episode in range(1, episodes + 1):
            state, _ = env.reset()

            episode_reward = 0

            for t in range(max_steps):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)

                action = actor(state_tensor).detach().numpy()[0] + NOISE_SCALE * noise.sample()

                next_state, reward, done, _, _ = env.step(action)

                # Store transition in the replay buffer
                replay_buffer.add((state, action, reward, next_state, float(done)))

                # Sample a random minibatch of transitions
                if len(replay_buffer.buffer) > BATCH_SIZE:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE, weighting = True, theta = THETA)

                    states = torch.FloatTensor(np.array(states))

                    actions = torch.FloatTensor(np.array(actions))
                    rewards = torch.FloatTensor(rewards).unsqueeze(1)
                    next_states = torch.FloatTensor(np.array(next_states))
                    dones = torch.FloatTensor(dones).unsqueeze(1)

                    # Update critic
                    target_actions = target_actor(next_states)
                    y = rewards + GAMMA * target_critic(next_states, target_actions) * (1 - dones)
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
                    soft_update(target_actor, actor, TAU)
                    soft_update(target_critic, critic, TAU)

                episode_reward += reward
                state = next_state

                if done:
                    break
            rewardsOut[episode -1] = episode_reward
            print(f"Episode {episode}, Reward: {episode_reward}")
            if episode % SAVE_EVERY == 0:
                # Save model every n-th episodes

                checkpoint = {
                    "model": actor.state_dict(),
                    "optimiser": actor_optimizer.state_dict(),
                }
                path = os.path.join(MODEL_SAVE_DIR, f"{ENV_NAME}_{episode}")
                torch.save(checkpoint, path)
                print(f"Model {episode} saved")

        # return rewards
        return np.mean(rewardsOut)
    return ddpg(env, episodes, max_steps)


def samplePopulation(populationSize):
    """
    GAMMA = hyperparameters["gamma"]
    TAU = hyperparameters["tau"]
    THETA = hyperparameters["theta"]
    ACTOR_LR = hyperparameters["actorLr"]
    CRITIC_LR = hyperparameters["criticLr"]
    NOISE_SCALE = hyperparameters["noise"]
    """
    pop = []
    for i in range(populationSize):
        gamma = torch.from_numpy(np.random.uniform(0,1,1)).float()
        tau = np.random.uniform(0,1,1)
        theta = np.random.uniform(0,10,1)
        actorLR = np.random.uniform(0.00001,0.01,1)[0]
        criticLR = np.random.uniform(0.00001, 0.01, 1)[0]
        noise = np.random.uniform(0, 0.5, 1)
        individual = {"gamma": gamma, "tau": tau,
                      "theta": theta, "actorLr": actorLR,
                      "criticLr": criticLR, "noise": noise}
        pop.append(individual)

    return pop



def mutatePop(pop, pMutate):
    def mutate(ind):
        params = ["gamma", "tau","theta","actorLR","criticLR", "noise"]
        helper = random.choice(params)
        if torch.is_tensor(ind[helper]):
            ind[helper] = ind[helper] + torch.from_numpy(np.random.uniform(0,1,1)) * ind[helper] * 2 # random amount of 2times previous parameter; heuristic; optimize here
            ind[helper] = ind[helper].float()
        else:
            ind[helper] = ind[helper] + np.random.uniform(0, 1, 1) * ind[helper] * 2
        return ind

    mut = np.random.randint(0, len(pop), math.ceil(len(pop) * pMutate))
    for ind in mut:
        pop[ind] = mutate(pop[ind])
    return pop

def evoOpt(nGenerations, popSize, pMutate):

    #init population
    pop = samplePopulation(popSize)
    print("population initialized")

    # start
    print("start Tournament")
    fitnessAvg = []
    for i in range(nGenerations):
        newPop = []
        fitness = np.zeros(popSize)
        for t in range(popSize):
            players = random.sample(pop, 4)
            leaderBoard = []
            for params in players:
                res = ddpgRun(params)
                leaderBoard.append(res)
            best = np.array(leaderBoard).argmax()
            newPop.append(players[best])
            fitness[t] = leaderBoard[best]
            print("winning fitness: ", leaderBoard[best])

        # update population
        print("generation", i, ", mean fitness: ", np.mean(np.array(fitness)))
        fitnessAvg.append(np.mean(np.array(fitness)))
        pop = mutatePop(newPop, pMutate)

    return [np.array(fitnessAvg), pop]


evoOpt(4, 5, 0.1)









