import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
import os

import math
import pandas as pd

from basic_ddpg.ddpg import DDPG


def samplePopulation(populationSize):
    pop = []
    for i in range(
        populationSize
    ):  ## switch with gaussian or log gaussian for parameter space above 0???
        gamma = torch.from_numpy(np.random.uniform(0, 1, 1)).float()
        tau = np.random.uniform(0, 1, 1)
        theta = np.random.uniform(0, 10, 1)
        actorLR = np.random.uniform(0.00001, 0.01, 1)[0]
        criticLR = np.random.uniform(0.00001, 0.01, 1)[0]
        noise = np.random.uniform(0, 0.5, 1)
        individual = {
            "gamma": gamma,
            "tau": tau,
            "theta": theta,
            "actorLr": actorLR,
            "criticLr": criticLR,
            "noise": noise,
        }
        pop.append(individual)

    return pop


def mutatePop(pop, pMutate):
    ##### we have to find good priors, not obvious from paper!!!!

    def mutate(ind):
        params = ["gamma", "tau", "theta", "actorLr", "criticLr", "noise"]
        helper = random.choice(params)

        if helper == "gamma" or helper == "tau" or helper == "noise":
            ind[helper] = ind[helper] + np.random.normal(0, 0.02)  ## add gaussian noise
            ind[helper] = np.clip(ind[helper], 0, 1)  # make sure valid range

        if helper == "theta":
            ind[helper] = ind[helper] + np.random.normal(0, 0.02)  ## add gaussian noise
            ind[helper] = np.clip(ind[helper], 0, np.Inf)  # make sure valid range

        if helper == "actorLr" or helper == "criticLr":
            ind[helper] = ind[helper] + np.random.normal(0, 0.001)  ## add gaussian noise
            ind[helper] = np.clip(ind[helper], 0, 1)
        return ind

    mut = np.random.randint(0, len(pop), math.ceil(len(pop) * pMutate))
    for ind in mut:
        pop[ind] = mutate(pop[ind])
    return pop


"""
test = samplePopulation(20)
for i in range(len(test)):
    print(mutatePop(test, 0.4))
"""


def evoOpt():
    ## init env
    ENV_NAME = "BipedalWalker-v3"
    MODEL_NAME = "evo_basic"

    EPISODES = 100
    MAX_STEPS = 100

    N_GENERATIONS = 100
    POP_SIZE = 100
    P_MUTATE = 0.1

    EVOLUTION_SAVE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "runs",
        ENV_NAME,
        MODEL_NAME,
    )

    env = gym.make(ENV_NAME, render_mode="rgb_array")

    DDPG(ENV_NAME, MODEL_NAME).init_test(env)  # Dummy writing and dir creation.

    # init population
    pop = samplePopulation(POP_SIZE)
    print("population initialized")

    # start
    print("start Tournament")
    fitnessAvg = []
    for i in range(N_GENERATIONS):
        newPop = []
        fitness = np.zeros(POP_SIZE)
        for t in range(POP_SIZE):
            players = random.sample(pop, 4)
            leaderBoard = []
            for params in players:
                ddpg = DDPG(
                    ENV_NAME=ENV_NAME,
                    MODEL_NAME=MODEL_NAME,
                    # General Hyperparameters
                    BUFFER_SIZE=1e6,
                    BATCH_SIZE=64,
                    GAMMA=params["gamma"],
                    TAU=params["tau"],
                    ACTOR_LR=params["actorLr"],
                    CRITIC_LR=params["criticLr"],
                    NOISE_SCALE=params["noise"],
                    # Episode Parameters
                    EPISODES=EPISODES,  # Number of episodes to run
                    MAX_STEPS=MAX_STEPS,  # Max number of steps per episode
                    SAVE_EVERY=np.Inf,  # Save every n-th model
                    VERBOSE=True,
                )
                res = ddpg.train(env)
                leaderBoard.append(res)
            best = np.array(leaderBoard).argmax()
            newPop.append(players[best])
            fitness[t] = leaderBoard[best]
            print("winning fitness: ", leaderBoard[best])

        # save results for each generation
        print("generation", i, ", mean fitness: ", np.mean(np.array(fitness)))

        def converter(x):
            """
            convert to correct format for csv
            """
            x = x.copy()
            x["gamma"] = x["gamma"].detach().cpu().item()
            x["tau"] = x["tau"][0]
            x["theta"] = x["theta"][0]
            x["noise"] = x["noise"][0]
            return x

        pop = list(map(lambda x: converter(x), pop))
        df = pd.DataFrame(pop)
        fitnessAvg.append(np.mean(np.array(fitness)))

        df.to_csv(os.path.join(EVOLUTION_SAVE_PATH, f"population{i}.csv"))
        np.savetxt(
            os.path.join(EVOLUTION_SAVE_PATH, f"fitness_population{i}.csv"),
            np.array(fitnessAvg),
            delimiter=",",
        )

        # update population
        pop = mutatePop(newPop, P_MUTATE)

    return [np.array(fitnessAvg), pop]


if __name__ == "__main__":
    evoOpt()
