"""
Test bench Class to seamlessly run different model, networks and optimization algorithms.

Author: Jules Kreuer / @not_a_feature
"""
from natsort import natsorted
import os
import shutil
import time
import torch
from matplotlib import pyplot as plt
from math import floor, ceil
import imageio
from PIL import Image


class TestBench:
    def init_test(self, env, overwrite=False):
        if overwrite:
            shutil.rmtree(self.MODEL_SAVE_DIR, ignore_errors=True)

        if os.path.isdir(self.MODEL_SAVE_DIR):
            raise FileExistsError(
                "Output of identical already present. Use overwrite=True to delete old run."
            )

        os.makedirs(self.MODEL_SAVE_DIR)

        header = f"""{self.MODEL_NAME}
Env              = {self.ENV_NAME}
Date             = {time.strftime('%H:%M %Z on %b %d, %Y')}
Model Path       = {self.MODEL_SAVE_DIR}
State dimension  = {env.observation_space.shape}
Action dimension = {env.action_space.shape}
--------------------------------------------
"""
        with open(self.GLOBAL_PATH, "w") as f:
            f.write(header)

            # Determine the max len for alignment
            max_length = max((len(var) for var in self.__dict__.keys()))
            # Write each variable and its value to the file
            for k, v in natsorted(self.__dict__.items()):
                f.write(f"{k.ljust(max_length)} = {v}\n")

        if os.path.isfile(self.LOSS_PATH):
            os.remove(self.LOSS_PATH)

        if self.VERBOSE:
            print(header)

    def log(self, path, l):
        with open(path, "a") as f:
            f.write(f"{l}\n")

        if self.VERBOSE:
            print(f"{l}")

    def save_model(self, actor, actor_optimizer, episode):
        checkpoint = {
            "model": actor.state_dict(),
            "optimiser": actor_optimizer.state_dict(),
        }
        path = os.path.join(self.MODEL_SAVE_DIR, f"model_e_{episode}")
        torch.save(checkpoint, path)

        if self.VERBOSE:
            print(f"Checkpoint {episode} saved.")

    def animate(self, env, start=0, end=None):
        models = self._get_saved_models()

        start = floor(start / self.SAVE_EVERY)
        if end is None:
            end = len(models)
        else:
            end = ceil(end / self.SAVE_EVERY)

        models = models[start:end]
        frames = []

        for i, model_path in enumerate(models, start=start):
            f = self._animate_episode(env, self.SAVE_EVERY * i, model_path)
            frames.extend(f)

        self._create_gif(frames, os.path.join(self.MODEL_SAVE_DIR, "animation.gif"))
        self._create_mkv(frames, os.path.join(self.MODEL_SAVE_DIR, "animation.mkv"))

    def _get_saved_models(self):
        models = os.listdir(self.MODEL_SAVE_DIR)
        models = [m for m in models if m.startswith("model_e_")]
        models = natsorted(models)
        models = [os.path.join(self.MODEL_SAVE_DIR, m) for m in models]
        return models

    def _create_gif(self, frames, output_file):
        frames = [Image.fromarray(frame) for frame in frames]
        imageio.mimsave(output_file, frames, format="GIF", duration=0.05)

    def _create_mkv(self, frames, output_file):
        writer = imageio.get_writer(
            output_file, fps=30, quality=9, codec="libx264", format="FFMPEG"
        )

        for frame in frames:
            writer.append_data(frame)

        writer.close()

    def _animate_episode(self, env, model_num, model_save_path):
        # Initialize actor network
        actor = self.load_actor(env, model_save_path)

        state = env.reset()[0]
        episode_reward = 0
        frames = []

        for t in range(self.MAX_STEPS):
            frame = env.render()
            frames.append(frame)

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = actor(state_tensor).detach().numpy()[0]
            next_state, reward, _, _, _ = env.step(action)

            episode_reward += reward
            state = next_state

        if self.VERBOSE:
            print(f"Animation {model_num}, Reward: {episode_reward:.3f}")

        return frames

    def plot_loss(self, thres=False):
        with open(self.LOSS_PATH, "r") as f:
            data = f.readlines()

        data = list(zip(*(l.strip().split(",") for l in data)))
        episode = [int(e) for e in data[0]]
        loss = [float(l) for l in data[1]]

        plt.plot(episode, loss, label="Reward")
        if thres:
            threshold = [float(l) for l in data[2]]
            plt.plot(episode, threshold, label="Threshold")

        plt.legend()
        plt.xlabel("Episode")
        plt.savefig(os.path.join(self.MODEL_SAVE_DIR, "loss.pdf"))

        if self.VERBOSE:
            plt.show()
        try:
            plt.clf()
            plt.close()
        except:
            pass
        return
