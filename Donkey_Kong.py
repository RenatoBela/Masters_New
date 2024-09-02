import os
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import ale_py
import shimmy

# Create a custom callback for rendering
class RenderCallback(BaseCallback):
    def __init__(self, env, render_freq=1000, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.env = env
        self.render_freq = render_freq
        self.step_counter = 0

    def _on_step(self) -> bool:
        self.step_counter += 1
        if self.step_counter % self.render_freq == 0:
            self.env.render(mode="human")
        return True

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
        self.previous_lives = 0
        self.previous_score = 0
        self.previous_position = None
        self.standing_time = 0

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.previous_lives = self.env.ale.lives()
        self.previous_score = 0
        self.previous_position = None
        self.standing_time = 0
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        current_lives = self.env.ale.lives()
        current_score = self.env.ale.getEpisodeFrameNumber()
        current_position = self.env.ale.getRAM()

        # Reward for going higher
        if self.previous_position is not None:
            if current_position[42] < self.previous_position[42]:  # Assuming 42 is the Y position in RAM
                reward += 10

        # Punishment for standing still
        if self.previous_position is not None:
            if np.array_equal(current_position, self.previous_position):
                self.standing_time += 1
            else:
                self.standing_time = 0

            if self.standing_time > 50:  # Adjust threshold as necessary
                reward -= 5

        self.previous_position = current_position

        # Reward for scoring
        if current_score > self.previous_score:
            reward += (current_score - self.previous_score) * 0.1

        self.previous_score = current_score

        return observation, reward, done, info

def train_model():
    # Create the environment
    env = make_atari_env("ALE/DonkeyKong-v5", n_envs=1)
    env = CustomRewardWrapper(env)
    env = DummyVecEnv([lambda: env])

    # Create the model
    model = PPO("CnnPolicy", env, verbose=1)

    # Train the model with the custom callback
    render_callback = RenderCallback(env, render_freq=1000)
    model.learn(total_timesteps=200000, callback=render_callback)

    # Create the directory if it doesn't exist
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model to the "models" folder
    model.save(os.path.join(model_dir, "ppo_donkeykong"))

    env.close()

def test_model():
    model_dir = "models"
    env = make_atari_env("ALE/DonkeyKong-v5", n_envs=1)
    env = CustomRewardWrapper(env)
    env = DummyVecEnv([lambda: env])

    # Load the model from the "models" folder
    model = PPO.load(os.path.join(model_dir, "ppo_donkeykong"))

    # Test the model and visualize
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render(mode="human")
        time.sleep(0.05)  # Render the environment to visualize the simulation

    env.close()

if __name__ == "__main__":
    train_model()
    test_model()
