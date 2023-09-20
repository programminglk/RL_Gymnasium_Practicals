import gymnasium as gym

env = gym.make("MiniGrid-LavaGapS7-v0", render_mode="human") # 5x5, 8x8, 16x16
# MiniGrid-Empty-5x5-v0
# MiniGrid-LavaGapS5-v0  S5, S6, S7
#

# look at the available minigrid environments here:
# https://minigrid.farama.org/environments/minigrid/

observation, info = env.reset(seed=42)

for _ in range(100):
#    action = policy(observation)  # User-defined policy function
   action = env.action_space.sample()

   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()