import gymnasium as gym
import numpy as np
import random                           #for choosing random actions
from collections import defaultdict
from minigrid.wrappers import FullyObsWrapper
import time
import matplotlib 
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

env=gym.make('MiniGrid-Empty-6x6-v0', render_mode='rgb_array')    #to train faster agent
test_env=gym.make('MiniGrid-Empty-6x6-v0', render_mode='human')   

env=FullyObsWrapper(env)
def get_state(env):
    base_env= env.unwrapped
    return base_env.agent_pos[0],base_env.agent_pos[1],base_env.agent_dir #gives a tuple of agent's position and direction
returns = defaultdict(list)                                
Q = defaultdict(lambda: np.zeros(env.action_space.n))       #store action value estimates for each action value pair
policy = {}    # make dictionary named policy (it store the best action for each state)
def generate_episode(env, policy, epsilon):
    episode = []   # to make a list of tuples (state, action, reward)
    obs, _ = env.reset()
    print("Initial Observation:", obs)
    state = get_state(env)
    done = False

    while not done:                                        #epsilon_greedy policy
        if state in policy and random.random() > epsilon:  #If the state is present in the policy and a random number is greater than epsilon
            action = policy[state]                         #then we use best known action else we use random action
        else:
            action = env.action_space.sample()             #random.random() this gives random number between 0 and 1

        _, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = get_state(env)

        episode.append((state, action, reward))            #store tuple (state,action ,reward) in list named episode
        state = next_state

    return episode
def monte_carlo_control(env, episodes=500, gamma=0.99):
    episode_rewards = []  # to store total rewards for each episode

    for i in range(episodes):
        epsilon=max(0.05, 1.0 - i/ episodes) #decay epsilon over time
        episode = generate_episode(env, policy,epsilon)
        #initialise G to 0 
        G = 0    
        total_reward=0                                
        for t in reversed(range(len(episode))):            #reverse the indices of list(episode)and iterate over time steps
            state, action, reward = episode[t]             #extract state, action and reward from the list
            G = gamma * G + reward
            total_reward += reward

                         
            returns[(state, action)].append(G)
            Q[state][action] = np.mean(returns[(state, action)]) #calculate the mean of retuns based on state and action
            policy[state] = int(np.argmax(Q[state]))            #update the policy with the action that has maximum Q value

        episode_rewards.append(total_reward)  # store total reward                    
    return policy, Q, episode_rewards

#to run the trained policy
def run_trained_policy(env, policy,max_steps=500):
    _, _ = env.reset()
    state = get_state(env)
    done = False
    steps = 0

    while not done and steps < max_steps:
        action = policy.get(state, env.action_space.sample())  # Use the trained policy or a random action
        _, _, terminated, truncated, _ = env.step(action)
        env.render()
        time.sleep(0.2)
        done = terminated or truncated
        state = get_state(env)
        steps += 1


trained_policy, Q_values,episode_rewards= monte_carlo_control(env, episodes=500, gamma=0.99)
run_trained_policy(test_env, trained_policy, max_steps=5000)

#plotting graphs
plt.plot(episode_rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.grid(True)
plt.show(block=True)