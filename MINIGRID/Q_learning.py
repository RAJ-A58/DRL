import numpy as np;
import gymnasium as gym
import time
from collections import defaultdict
from minigrid.wrappers import FullyObsWrapper
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
env=gym.make('MiniGrid-Empty-6x6-v0', render_mode='rgb_array')
visible_env=gym.make('MiniGrid-Empty-6x6-v0', render_mode='human')
env=FullyObsWrapper(env)
alpha=0.9           # learning rate
gamma=0.95              # discount factor
epsilon=1.0
epsilon_decay=0.9995
epsilon_min=0.01
max_episodes=5000    #maximum number of episodes
max_steps=500        #maxium number of steps per episode

Q=defaultdict(lambda: np.zeros(env.action_space.n))     #initialise Q table with zeros
episode_rewards=[]                                      #forming a list named episode_reward

def choose_action(state):
    if np.random.random() < epsilon:
        return env.action_space.sample()  # explore
    else:
        return np.argmax(Q[state])  # exploit (epsilon-greedy policy)
    
def get_state(env):
    position= tuple(env.unwrapped.agent_pos)
    direction= env.unwrapped.agent_dir
    return (position, direction)  #returns a tuple of agent's position and direction

#q-learning algorithm
for episode in range(max_episodes):
    obs,state=env.reset()
    state=get_state(env)
    total_reward=0
    done=False

    while not done:
        action=choose_action(state)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state=get_state(env)

        best_next_action=np.argmax(Q[next_state])                       #Q-learning policy
        td_error=reward + gamma * Q[next_state][best_next_action] - Q[state][action]       #calulation of TD error 
        Q[state][action] += alpha * td_error

        state=next_state
        total_reward += reward
        done = terminated or truncated

    episode_rewards.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)  # decay epsilon

# Run the trained policy
def run_trained_policy(visible_env, max_steps=500):
    _, _ = visible_env.reset()
    state = get_state(visible_env)
    done = False
    steps = 0
    total_reward = 0

    while not done and steps < max_steps:
        action = choose_action(state)  # choose action based on the learned policy
        obs, reward, terminated, truncated, info = visible_env.step(action)
        next_state = get_state(visible_env)
        time.sleep(0.5)
        total_reward += reward
        state = next_state
        done = terminated or truncated
        steps += 1

    print(f"Total Reward: {total_reward}")
    visible_env.close()                      #close the environment.

run_trained_policy(visible_env)

plt.plot(episode_rewards, label='Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Total Reward') 
plt.title('Q-Learning policy')
plt.grid(True)
plt.legend()
plt.show()