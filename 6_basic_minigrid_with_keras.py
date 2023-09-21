import gymnasium as gym

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

env = gym.make("MiniGrid-LavaGapS7-v0", render_mode="human") # 5x5, 8x8, 16x16
# MiniGrid-Empty-5x5-v0
# MiniGrid-LavaGapS5-v0  S5, S6, S7

# look at the available minigrid environments here:
# https://minigrid.farama.org/environments/minigrid/


gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
eps = np.finfo(np.float32).eps.item()  # Epsilone - Smallest number such that 1.0 + eps != 1.0 

num_inputs = 4
num_actions = 2
num_hidden = 128

# Using Keras __ API to build the model
inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])


observation, info = env.reset(seed=42)

for _ in range(100):
#    action = policy(observation)  # User-defined policy function
   action = env.action_space.sample()

   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()