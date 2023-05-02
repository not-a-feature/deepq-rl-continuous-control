"""
Test bench to seamlessly run a model repeatedly and save the performance.

Author: Jules Kreuer / @not_a_feature
"""

import gymnasium as gym
from stratified_ddpg.ddpg import DDPG

if __name__ == "__main__":
    ENV_NAME = "Pendulum-v1"

    for i in range(0, 10):
        MODEL_NAME = f"adapt_strat_sb_neg_{i}"

        env = gym.make(ENV_NAME, render_mode="rgb_array")
        ddpg = DDPG(
            ENV_NAME,
            MODEL_NAME,
            BUFFER_SIZE=10000,
            REWARD_THRESHOLD=-0.5,
            STRATIFY_RATIO=0.5,
            EPISODES=200,
            VERBOSE=True,
        )

        ddpg.init_test(env, overwrite=False)
        ddpg.train(env)
        ddpg.plot_loss(thres=True)
        ddpg.animate(env)
        del ddpg
        del env
