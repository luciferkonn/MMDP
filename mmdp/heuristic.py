import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from grid_city import GridWorld

ROOT = '/home/lucifer/Documents/Git/MMDP/'


def get_q_value(filename=ROOT+'data/one_agent_q_value.txt'):
    prob = np.loadtxt(filename)
    return prob.reshape(25, 5)


# Map coarse grid to finer grid
def map_location(x, y, pool_prob, cust_prob):
    temp = 100
    for i in range(5):
        for j in range(5):
            minus = np.abs(cust_prob[x, y] - pool_prob[i, j])
            if minus < temp:
                temp = minus
                new_x = i
                new_y = j
    return new_x, new_y


def heuristic(args, env):
    # hyper-parameters
    episode_len = args.episode_len
    epsilon = args.epsilon
    runs = args.runs
    grid_size = args.grid_size
    n_actions = args.n_actions
    n_agents = args.n_agents
    # total reward
    total_reward = np.zeros(episode_len)
    for j in range(runs):
        print('RUNS:{}'.format(j))
        # initialization
        q_value = get_q_value()
        # loop for each episode
        for i in range(episode_len):
            # initialize each agent's state
            states, done = env.reset()
            total = 0
            action = [0] * len(states)
            action_others = [0] * len(action)
            # loop for every state of agent
            while not done[0]:
                for j, state in enumerate(states):
                    (x, y) = state['loc']
                    time = state['time']
                    id = state['id']
                    # map location
                    (map_x, map_y) = map_location(x, y, env.pool_prob, env.cust_prob)
                    # eplison-greedy choose acion
                    if np.random.binomial(1, epsilon) == 1:
                        action[j] = np.random.randint(n_actions)
                    else:
                        action[j] = np.argmax(q_value[map_x*5+map_y, :])
                        # print(action[j])
                # every agent take a step
                next_state, reward, done, info = env.step(action)
                states = env.get_states()
                for k, _ in enumerate(reward):
                    total += reward[k]
            total_reward[i] += total
            if i % 10 == 0:
                print('Episode:{}, reward{}'.format(i, total))
    return total_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-agent DDPG')
    # add argument
    parser.add_argument('--grid_size', default=100, type=int, help='the size of a grid world')
    parser.add_argument('--n_actions', default=5, type=int, help='total number of actions an agent can take')
    parser.add_argument('--filename', default=ROOT + '/data/pr.txt', type=str, help='Pick-up probability file')
    parser.add_argument('--n_agents', default=1, type=int, help='the number of agent play in the environment')
    parser.add_argument('--runs', default=30, type=int, help='the number of times run the game')
    parser.add_argument('--episode_len', default=300, type=int, help='number of episode')
    parser.add_argument('--epsilon', default=0.1, type=int, help='epsilon greedy')
    # parser args
    args = parser.parse_args()
    env = GridWorld(args=args, terminal_time=1000, reward_stay=-1, reward_hitwall=-2, reward_move=-1, reward_pick=2)
    # print(get_q_value())
    print(env.cust_prob)
    print(env.pool_prob)
    heuristic(args, env)
