import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from grid_city import GridWorld

ROOT = '/home/lucifer/Documents/Git/MMDP/'


def mqlearning(args, env, episode_len=1000, learning_rate=0.5, epsilon=0.1, gamma=1, runs=30):
    grid_size = args.grid_size
    n_actions = args.n_actions
    # initialization
    q_value = np.zeros((grid_size*grid_size, n_actions))
    # total reward
    total_reward = np.zeros(episode_len)

    for _ in range(runs):
    # loop for each episode
        for i in range(episode_len):
            # initialize each agent's state
            states, done = env.reset()
            action = [0] * len(states)
            # loop for every state of agent
            while not done[0]:
                for j, state in enumerate(states):
                    (x, y) = state['loc']
                    time = state['time']
                    id = state['id']
                    # eplison-greedy choose acion
                    if np.random.binomial(1, epsilon) == 1:
                        action[j] = np.random.randint(n_actions)
                    else:
                        action[j] = np.argmax(q_value[x*grid_size+y])
                        # print(action[j])
                # every agent take a step
                next_state, reward, done, info = env.step(action)

                # updata q-value simultaneously
                for k, _ in enumerate(action):
                    (next_x, next_y) = next_state[k]['loc']
                    q_value[x*grid_size+y, action[k]] += learning_rate * (reward[k] + gamma * np.max(q_value[next_x *
                                                            grid_size + next_y, :]) - q_value[x*grid_size+y, action[k]])
                    (x, y) = (next_x, next_y)
                    total_reward[i] += reward[k]
            if i % 100 == 0:
                print('Episode:{}, reward{}'.format(i, total_reward[i]))
    return total_reward, q_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-agent DDPG')
    # add argument
    parser.add_argument('--grid_size', default=6, type=int, help='the size of a grid world')
    parser.add_argument('--n_actions', default=5, type=int, help='total number of actions an agent can take')
    parser.add_argument('--filename', default=ROOT+'/data/pr.txt', type=str, help='Pick-up probability file')
    parser.add_argument('--n_agents', default=1, type=int, help='the number of agent play in the environment')

    # parser args
    args = parser.parse_args()
    env = GridWorld(args=args, terminal_time=1000, reward_stay=-1, reward_hitwall=-1, reward_move=-1)
    rewards, q_value = mqlearning(args=args, env=env, episode_len=1000, runs=1)
    rewards /= 20
    print(q_value.reshape(args.grid_size*args.grid_size, 5))
    plt.figure(1)
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('sum of rewards during episode')
    plt.legend()
    plt.show()