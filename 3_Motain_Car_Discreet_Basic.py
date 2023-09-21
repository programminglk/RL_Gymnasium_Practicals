import gymnasium as gym
env = gym.make("MountainCar-v0", render_mode="human") # CartPole-v1, MountainCar-v0, CliffWalking-v0, LunarLander-v2
observation, info = env.reset()

import time

print(gym.envs.registry.keys())

for _ in range(100):

    action = env.action_space.sample() # <--this is random policy. (Add your policy function here)

    print(f"action selected at {_} is: {action}")  

    observation, reward, terminated, truncated, info = env.step(action)

    print(f"observation at {_} is: {observation}") 
    print(f"reward at {_} is: {reward}")
    print("-----------------------------------\n")

    if terminated or truncated:
        observation, info = env.reset()

        time.sleep(0.1)


env.close()