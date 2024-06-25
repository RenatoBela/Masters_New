import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create the environment
env = make_vec_env("CarRacing-v2", n_envs=1)

# Create the model
model = PPO("CnnPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=5000)

# Create the directory if it doesn't exist
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the model to the "models" folder
model.save(os.path.join(model_dir, "ppo_carracing2"))

# To load the model from the "models" folder
model = PPO.load(os.path.join(model_dir, "ppo_carracing2"))

# Test the model and visualize
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render(mode="human")  # Render the environment to visualize the simulation

env.close()
