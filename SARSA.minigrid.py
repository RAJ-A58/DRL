import gymnasium as gym
import numpy as np
from collections import defaultdict
from minigrid.wrappers import FullyObsWrapper
import random
import matplotlib
matplotlib.use('TkAgg') 
import time
import matplotlib.pyplot as plt

env=gym.make('MiniGrid-Empty-6x6-v0', render_mode='rgb_array')
test_env=gym.make('MiniGrid-Empty-6x6-v0', render_mode='human')
env=FullyObsWrapper(env)
alpha = 0.1     # learning rate
lambda_ = 0.8   # decay rate    
gamma = 0.9     # discount factor
epsilon = 0.3   # exploration rate
num_episodes = 1000
actions=env.action_space.n

def get_state(env):
    agent_pos = tuple(env.unwrapped.agent_pos)
    agent_dir = env.unwrapped.agent_dir
    return (agent_pos, agent_dir)              #return a tuple (x,y) cordinates of the agent and its direction.
            
def choose_action(state,Q):
    if random.random() < epsilon:          # random.random( generates a random float between 0 and 1)
        return env.action_space.sample()  # explore with probability epsilon.  
    else:
        return np.argmax(Q[state])  # exploit
    
Q= defaultdict(lambda: np.zeros(actions))  # action-value function
E= defaultdict(lambda: np.zeros(actions))  # eligibility traces initialised to zero
all_episode_rewards = []  # to store rewards for each episode

for episode in range(num_episodes):
    obs, info = env.reset()
    state = get_state(env)
    action=choose_action(state, Q)

    E.clear()   #cleat the eligibility traces at the start of each episode

    done=False
    total_reward = 0

    while not done:
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = get_state(env)
        next_action = choose_action(next_state, Q)

        td_error = reward + gamma *Q[next_state][next_action] - Q[state][action]
        E[state][action] += 1

        for s in Q:
            for a in range(actions):
                Q[s][a] += alpha * td_error * E[s][a]
                E[s][a] *= gamma * lambda_

        state = next_state
        action = next_action    
        total_reward += reward
        done = terminated or truncated
    all_episode_rewards.append(total_reward)
env.close()

def greedy_action(state, Q):
    return np.argmax(Q[state])  # choose the action with the highest Q-value for the current state

#run the trained policy
def run_trained_policy(test_env, max_steps=500):
    _, _ = test_env.reset()
    state = get_state(test_env)
    done = False
    steps = 0
    total_reward = 0

    while not done and steps < max_steps:
        action = greedy_action(state, Q)  # use learned policy
        _, reward, terminated, truncated, _ = test_env.step(action)
        next_state = get_state(test_env)

        total_reward += reward
        state = next_state
        done = terminated or truncated
        steps += 1

    print(f"Total Reward: {total_reward}")
    test_env.close()

run_trained_policy(test_env)


# Plot total rewards over episodes
plt.plot(all_episode_rewards, label='Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('SARSA(λ) on MiniGrid')
plt.grid(True)
plt.legend()
plt.show()