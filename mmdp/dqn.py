import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from grid_city import GridWorld
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import namedtuple
import random
import numpy as np

ROOT = '/home/lucifer/Documents/Git/MMDP/'

# if GPU is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000
N_ACTIONS = 5
N_STATES = 100*100  # grid_size * grid_size


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = Variable(x)
        x = F.relu(self.fc1(x))
        return self.out(x)


class DQN(object):

    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        print(x.view)
        # input only one sample
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            # action = action[0] if ENV
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gater(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1) # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


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

    # Create a network
    dqn = DQN()

    # Evaluating......
    print('\nCollecting experience...')
    for i_episode in range(400):
        states, done = env.reset()
        ep_r = 0
        while True:
            (x, y) = states[0]['loc']
            print(x, y)
            a = dqn.choose_action(x*args.grid_size+y)

            # take an action return next_state, reward, done, info
            torch.unsqueeze(a)
            s_, r, done, info = env.step(a)

            # modify the reward
            # x, x_dot, theta, theta_dot = s_

            dqn.store_transition(s, a, r, s_)

            ep_r += r
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print('Ep:', i_episode,
                          '| Ep_r:', round(ep_r, 2))

            if done:
                break
            s = s_





