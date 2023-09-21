import gymnasium as gym

import time

env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human") # 5x5, 8x8, 16x16
# MiniGrid-Empty-5x5-v0
# MiniGrid-LavaGapS5-v0  S5, S6, S7
#

# look at the available minigrid environments here:
# https://minigrid.farama.org/environments/minigrid/

observation, info = env.reset(seed=42)

for i in range(2):
#    action = policy(observation)  # User-defined policy function
   action = env.action_space.sample()
   print(f"iteration {i} action is: {action}")

   observation, reward, terminated, truncated, info = env.step(action)

   print(f"iteration {i} observation is: {observation} \n")

   if terminated or truncated:  
      if i == 1:
         time.sleep(360)
         print("sleeping")
      observation, info = env.reset()
   
   if i == 1:
      time.sleep(360)
      print("sleeping")


env.close()