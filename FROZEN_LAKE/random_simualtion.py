import gymnasium as gym
env = gym.make("FrozenLake-v1", render_mode="human", desc=None, map_name="4x4", is_slippery=True)
state, info= env.reset()
for _ in range(100):
    random_action = env.action_space.sample()
    state, reward, done, _, info = env.step(random_action)
    if done:
        break
env.close()