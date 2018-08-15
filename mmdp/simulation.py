import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from grid_city import GridWorld

ROOT = '/home/lucifer/Documents/Git/MMDP/'


def simulation(args, env, episode_len=1000, learning_rate=0.9, epsilon=1, gamma=1, runs=30):
    grid_size = args.grid_size
    n_actions = args.n_actions
    n_agents = args.n_agents

    # total reward
    total_reward = np.zeros(episode_len)
    global_total = 0
    for j in range(runs):
        print('RUNS:{}'.format(j))
        # initialization
        q_value = np.zeros((grid_size*grid_size, n_actions, n_actions**(n_agents-1)))

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
                    # eplison-greedy choose acion
                    if np.random.binomial(1, epsilon) == 1:
                        action[j] = np.random.randint(n_actions)
                    else:
                        action[j] = np.argmax(q_value[x*grid_size+y,:,action_others[j]])
                        # print(action[j])
                # every agent take a step
                next_state, reward, done, info = env.step(action)

                # find the others actions
                for k, _ in enumerate(action):
                    # implement agent others
                    action_oth = np.delete(action, k, 0)
                    action_others[k] = 0
                    size = len(action_oth)
                    for oth, _ in enumerate(action_oth):
                        if oth == len(action_oth) - 1:
                            action_others[k] += action_oth[oth]
                        else:
                            action_others[k] += action_oth[oth] * n_actions**(size - oth - 1)


                # updata q-value simultaneously
                for k, _ in enumerate(action):
                    # print(info['n'][k]['event'])
                    (next_x, next_y) = next_state[k]['loc']

                    q_value[x*grid_size+y, action[k], action_others[k]] += learning_rate * (reward[k] + gamma * np.max(q_value[next_x *
                                                            grid_size + next_y, :, action_others[k]]) - q_value[x*grid_size+y, action[k], action_others[k]])
                    (x, y) = (next_x, next_y)
                    total += reward[k]
                states = env.get_states()
            total_reward[i] += total
            total_passenger = env.num_pick
            if i % 10 == 0:
                print('Episode:{}, reward{}, passenger{}'.format(i, total, total_passenger))
            global_total += total
            print('Global reward:' + str(global_total))
    print('Average:' + str(global_total / 30000))
    return total_reward, q_value

def printOptimalPolicy(q_value):
    z = []
    for i in range(6):
        z.append([])
        for j in range(6):
            y = np.argmax(q_value[i * 6 + j, :])
            if y == 0:
                z[-1].append('STAY')
            elif y == 1:
                z[-1].append(' UP ')
            elif y == 2:
                z[-1].append('DOWN')
            elif y == 3:
                z[-1].append('LEFT')
            elif y == 4:
                z[-1].append('RIGT')
    for row in z:
        print(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-agent DDPG')
    # add argument
    parser.add_argument('--grid_size', default=100, type=int, help='the size of a grid world')
    parser.add_argument('--n_actions', default=5, type=int, help='total number of actions an agent can take')
    parser.add_argument('--filename', default=ROOT+'/data/pr.txt', type=str, help='Pick-up probability file')
    parser.add_argument('--n_agents', default=1, type=int, help='the number of agent play in the environment')
    parser.add_argument('--runs', default=1, type=int, help='the number of times run the game')

    # parser args
    args = parser.parse_args()
    env = GridWorld(args=args, terminal_time=1000, reward_stay=-1, reward_hitwall=-2, reward_move=-1, reward_pick=2, aggre=False)
    rewards, q_value = simulation(args=args, env=env, episode_len=30000, runs=args.runs)
    rewards /= args.runs
    print(str(rewards))

    # plt.figure(1)
    # plt.plot(rewards)
    # plt.xlabel('Episodes')
    # plt.ylabel('sum of rewards during episode')
    # plt.legend()
    # plt.show()
    # printOptimalPolicy(q_value)
