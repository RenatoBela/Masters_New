import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
import pygame 
import numpy as np

def get_user_action():
    keys = pygame.key.get_pressed()  # Get the state of all keyboard keys
    action = np.array([0.0, 0.0, 0.0])  # Default action [steering, acceleration, brake]

    # Steering
    if keys[pygame.K_LEFT]:
        action[0] = -1.0  # Steer left
    elif keys[pygame.K_RIGHT]:
        action[0] = 1.0  # Steer right

    # Acceleration
    if keys[pygame.K_UP]:
        action[1] = 1.0  # Accelerate

    # Brake
    if keys[pygame.K_DOWN]:
        action[2] = 0.8  # Apply brake

    return action

pygame.init()
# Create the environment
env = make_vec_env("CarRacing-v2", n_envs=1)

# Create a TensorBoard log directory
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Define the TensorBoard callback
class TensorBoardCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(TensorBoardCallback, self).__init__(verbose)
        self.log_dir = log_dir

    def _on_step(self) -> bool:
        # Log training metrics
        self.logger.dump(self.num_timesteps)
        return True

# Create the model
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)

# Train the model
#model.learn(total_timesteps=5000, callback=TensorBoardCallback(log_dir))

# Save the model to the "models" folder
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save(os.path.join(model_dir, "ppo_carracing2"))

# To load the model from the "models" folder
model = PPO.load(os.path.join(model_dir, "ppo_carracing2"))

# Test the model and visualize
obs = env.reset()
for _ in range(1000):
    #action, _states = model.predict(obs, deterministic=True)
    action = get_user_action()
    obs, rewards, dones, info = env.step(action)
    env.render(mode="human")  # Render the environment to visualize the simulation

env.close()
