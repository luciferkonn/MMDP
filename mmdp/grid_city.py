import gym
import numpy as np
import os
ROOT = '/home/lucifer/Documents/Git/MMDP/'

# read the customer probability from the file
from car import Car


def prob_read(filename):
    data = np.loadtxt(fname=filename, dtype=np.float32)
    return data


class GridWorld(gym.Env):

    def __init__(self, args, reward_hitwall=-0.2, reward_collision=-0.1, reward_pick=1, reward_stay=-0.1, reward_move=-0.1,
                 threshold_num=50000000, terminal_time=1000):
        self.grid_size = args.grid_size
        self.n_agents = args.n_agents
#         self.cust_prob = self.prob_set(args.filename)
        self.cust_prob = [[0.34114716 0.35664167 0.39854643 0.40897067 0.45349451]
                         [0.33441988 0.31124772 0.34391435 0.34767191 0.38524335]
                         [0.35419336 0.31008612 0.32275275 0.32408603 0.35054635]
                         [0.37512068 0.32434676 0.31939436 0.3037686  0.32737175]
                         [0.4023588  0.30629816 0.29801244 0.28048192 0.3314184 ]]
        self.agents = self.init_agents(self.n_agents)
        self.grid = np.zeros((self.grid_size, self.grid_size))
        # set time for the agent, start at 0
        self.time = 0
        self.terminal_time = terminal_time
        # reward for all kinds of event
        self.reward_hitwall = reward_hitwall
        self.reward_collision = reward_collision
        self.reward_pick = reward_pick
        self.reward_stay = reward_stay
        self.reward_move = reward_move
        # threshold for control the maximum cars in each grid
        self.threshold_num = threshold_num
        self.threshold = np.zeros((self.grid_size, self.grid_size))

    def step(self, action):
        # set state, reward, done, info for every agent
        state_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        # time add 1
        self.time += 1

        self.make_move(action)
        # return for each agent
        for i, agent in enumerate(self.agents):
            # next_state, reward, done, info for each agent
            state_n.append(self.get_state(agent))
            reward_n.append(self.get_reward(agent))
            done_n.append(self.get_done(agent))
            info_n['n'].append(self.get_info(agent))
        return state_n, reward_n, done_n, info_n

    # reset each agent and return their state
    def reset(self):
        self.agents = self.init_agents(self.n_agents)
        self.time = 0
        self.threshold = np.zeros((self.grid_size, self.grid_size))
        states = []
        for agent in self.agents:
            states.append(self.get_state(agent))
        done = [False]
        return states, done

    def get_states(self):
        state_n = []
        for i, agent in enumerate(self.agents):
            state_n.append(self.get_state(agent))
        return state_n

    def init_agents(self, n_agents):
        agents = []
        for i in range(0, self.n_agents):
            rand_loc = self.agent_random_loc()
            car = Car(rand_loc, i)
            agents.append(car)
        return agents

    # return a random loc for an agent
    def agent_random_loc(self):
        rand_loc = self.rand_pair(0, self.grid_size)
        return rand_loc
#         return 5, 4
    # generate a random loc for x and y
    @staticmethod
    def rand_pair(s, grid_size):
        return np.random.randint(s, grid_size), np.random.randint(s, grid_size)

    # get state(time, location, id)
    def get_state(self, agent):
        state = {'time': self.time, 'loc': agent.loc, 'id': agent.id}
        return state

    def get_reward(self, agent):
        return agent.reward

    def get_done(self, agent):
        time = self.get_state(agent)['time']
        if self.time == self.terminal_time:
            return True
        else:
            return False

    def get_info(self, agent):
        return agent.step

    # make a move for every agent
    def make_move(self, action):
        car_locations = self.get_car_locs()  # save for collision use
        for i, agent in enumerate(self.agents):
            (x, y) = self.convert_action_to_loc(agent, action[i])

            # out of the grid
            if x not in range(self.grid_size) or y not in range(self.grid_size):
                agent.reward = self.reward_hitwall
                agent.step['event'] = 'hit_wall'

            # pick up customer
            else:
                if action == 0:
                    agent.step['event'] = 'stay'
                    agent.reward = self.reward_stay
                else:
                    if (x, y) != agent.loc:
                        self.threshold[x, y] += 1
                    if self.threshold[x, y] <= self.threshold_num:
                        if self.can_pick(x, y):
                            agent.reward = self.reward_pick
                            agent.step['event'] = 'pick'
                        else:
                            agent.reward = self.reward_move
                            agent.step['event'] = 'idle'
                    else:
                        agent.reward = self.reward_move
                        agent.step['event'] = 'move'
                    agent.loc = (x, y)

    # take an intended action
    @staticmethod
    def convert_action_to_loc(agent, action):
        curr_loc = agent.loc
        new_loc = curr_loc
        if action == 0:  # stay
            pass
        elif action == 3:  # left
            new_loc = (curr_loc[0], curr_loc[1] - 1)
        elif action == 1:  # up
            new_loc = (curr_loc[0] - 1, curr_loc[1])
        elif action == 4:  # right
            new_loc = (curr_loc[0], curr_loc[1] + 1)
        elif action == 2:  # down
            new_loc = (curr_loc[0]+1, curr_loc[1])
        else:
            print("ERROR on INTENDED ACTION: Returning to current location")
        return new_loc

    # get all cars locations
    def get_car_locs(self):
        locs = []
        for car in self.agents:
            locs.append(car.loc)
        return locs

    # check if a car agent can find a passenger at location(x, y)
    def can_pick(self, x, y):
        return np.random.binomial(1, self.cust_prob[x, y])

    # give every grid(x, y) a pick up probability from pr.txt
    def prob_set(self, filename):
        data = prob_read(filename)
        prob = np.zeros((self.grid_size, self.grid_size)).reshape(-1, 1)
        j = 0
        for i in range(np.size(data)):
            if data[i] >= 1:
                continue
#                 data[i] = 1
#             else:
#                 data[i] = 0
            j += 1
            if j >= self.grid_size * self.grid_size:
                break
            prob[j] = data[i]
        print(prob.reshape(self.grid_size, self.grid_size))
        return prob.reshape(self.grid_size, self.grid_size)
