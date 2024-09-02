import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

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
model_baseline = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)

model_1 = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    n_steps=4096,
    batch_size=128
)

# Primjer za postavku 3
model_2 = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=0.0001,
    clip_range=0.1
)

# Primjer za postavku 4
model_3 = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    gamma=0.995,
    gae_lambda=0.98
)

# Primjer za postavku 5
model_4 = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    ent_coef=0.01
)
# Train the model
model_1.learn(total_timesteps=5000, callback=TensorBoardCallback(log_dir))

# Save the model to the "models" folder
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_1.save(os.path.join(model_dir, "ppo_carracing2"))

# To load the model from the "models" folder
model_loaded = PPO.load(os.path.join(model_dir, "ppo_carracing2"))

# Test the model and visualize
obs = env.reset()
for _ in range(1000):
    action, _states = model_loaded.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render(mode="human")  # Render the environment to visualize the simulation

env.close()
