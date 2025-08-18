import gymnasium as gym
import numpy as np
import time
from collections import defaultdict
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for matplotlib
import matplotlib.pyplot as plt
from minigrid.wrappers import FullyObsWrapper
import random

#initialise both training and test environments
env=gym.make('MiniGrid-Empty-6x6-v0', render_mode='rgb_array')
test_env=gym.make('MiniGrid-Empty-6x6-v0', render_mode='human')
env=FullyObsWrapper(env)  # wrap the environment for full observation

#initilising the parameters
alpha = 0.1     # learning rate
lambda_ = 0.6   # decay rate
gamma = 0.99 # discount factor
epsilon = 0.2   # exploration rate
epsilon_decay = 0.995
epsilon_min = 0.05  # minimum exploration rate
num_episodes = 1000

#initialise the q table and eligibility trace
Q= defaultdict(lambda: np.zeros(env.action_space.n))
E= defaultdict(lambda: np.zeros(env.action_space.n))

all_episode_rewards = []  # to store rewards for each episode

#Function to choose an action
def choose_action(state):
    if random.random()<epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

def get_state(env):
    base_env=env.unwrapped
    position = tuple(base_env.agent_pos)
    direction = base_env.agent_dir
    return (position, direction)  # returns a tuple of agent's position and direction

for episode in range(num_episodes):
    env.reset()
    state=get_state(env)
    action=choose_action(state)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)  # decay epsilon after each episode

    done = False
    total_reward = 0
    

    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        next_state = get_state(env)
        next_action = choose_action(next_state)

        td_error = reward + gamma * Q[next_state][next_action] - Q[state][action]
        E[state][action] += 1

        for s in Q:
            for a in range(env.action_space.n):
                Q[s][a] += alpha * td_error * E[s][a]
                E[s][a] *= gamma * lambda_

        state = next_state
        action = next_action
        total_reward += reward
        if terminated or truncated:
            done = True
    all_episode_rewards.append(total_reward)

        
env.close()  # close the environment after each episode

def greedy_action(state, Q):
    return np.argmax(Q[state])  # choose the action with the highest Q-value

def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards per Episode')
    plt.show()

def run_trained_policy(test_env, max_steps=500):
    _, _ = test_env.reset()
    state = get_state(test_env)
    done = False
    total_reward = 0

    for step in range(max_steps):
        action = greedy_action(state, Q)
        obs, reward, terminated, truncated, info = test_env.step(action)
        next_state = get_state(test_env)

        total_reward += reward
        state = next_state

        if terminated or truncated:
            done = True
            break

    print(f"Total Reward: {total_reward}")
    test_env.close()

run_trained_policy(test_env)
plot_rewards(all_episode_rewards) 