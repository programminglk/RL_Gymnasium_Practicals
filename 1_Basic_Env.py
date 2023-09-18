import gymnasium as gym
env = gym.make("CliffWalking-v0", render_mode="human") # CartPole-v1, MountainCar-v0, CliffWalking-v0, LunarLander-v2
observation, info = env.reset()

print(gym.envs.registry.keys())

for _ in range(200):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()