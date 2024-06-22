import gymnasium as gym
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras.optimizer_v2.adam import Adam


from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

#import os
#os.environ["TF_USE_LEGACY_KERAS"]="1"

# Create the environment
env= gym.make("CarRacing-v2")
observation, info = env.reset(seed=42)
input_shape = env.observation_space.shape
states = env.observation_space.shape[0]
actions = env.action_space.shape[0]
print(input_shape)
print(states)
print(actions)

# Define the model
model = Sequential()
model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(actions, activation='linear'))

# Define the agent
agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.001
)

# Compile the agent
agent.compile(Adam(learning_rate= 0.01), metrics=['mae'])

# Fit the agent
agent.fit(env, nb_steps=100000, visualize=False, verbose=1)

# Test the agent
results = agent.test(env, nb_episodes=10, visualize=True)
print(np.mean(results.history["episode_reward"]))

env.close()