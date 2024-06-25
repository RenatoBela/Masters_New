import os
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import BaseCallback
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

def train_model():
    # Create the environment
    env = make_atari_env("ALE/DonkeyKong-v5", n_envs=2)

    # Create the model
    model = PPO("CnnPolicy", env, verbose=1)

    # Train the model with the custom callback
    render_callback = RenderCallback(env, render_freq=1000)
    model.learn(total_timesteps=200000)

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
    print("train")
    # Load the model from the "models" folder
    model = PPO.load(os.path.join(model_dir, "ppo_donkeykong"))
    print("model loaded")
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
