import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from grid_city import GridWorld
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import namedtuple
import random
import numpy as np
from replay_memory import ReplayMemory

sys.path.append("../")
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

    def __init__(self, memory=None):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.replay = memory

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # print(x.view)
        # input only one sample
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = np.array(torch.max(actions_value, 1)[1].data.numpy()).astype(int)
            # action = action[0] if ENV
        else:
            action = np.array([np.random.randint(0, N_ACTIONS)]).astype(int)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        # transition = np.hstack((state, [action, reward], next_state))
        # index = self.memory_counter % MEMORY_CAPACITY
        # self.memory[index, :] = transition
        self.memory_counter += 1
        memory = {'state': state, 'action': action, 'reward': reward, 'new_state': next_state}
        self.replay.addToMemory(memory, done)

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.replay.getMinibatch()
        b_s = b_memory[0][0][0]["loc"]
        (l1, l2) = b_s
        b_s = torch.FloatTensor(100 * 100)
        b_s[l1*args.grid_size+l2] = 1
        # b_s = b_memory[0][0]
        # print(b_memory[0][1])
        # print(b_memory[0])
        b_a = torch.LongTensor(b_memory[0][1])
        b_r = torch.FloatTensor(b_memory[0][2])
        b_s_ = b_memory[0][3][0]["loc"]
        (l3, l4) = b_s_
        b_s_ = torch.FloatTensor(100 * 100)
        b_s_[l3 * args.grid_size + l4] = 1

        # q_eval w.r.t the action in experience
        q_eval = torch.unsqueeze(self.eval_net(b_s), 0).gather(1, torch.unsqueeze(b_a, 0))  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't back propagate
        q_target = b_r + GAMMA * torch.unsqueeze(q_next, 0).max(1)[0] # .view(BATCH_SIZE, 1)  # shape (batch, 1)
        # q_target = b_r + GAMMA * q_next
        loss = self.loss_func(q_eval[0], q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-agent DDPG')
    # add argument
    parser.add_argument('--grid_size', default=100, type=int, help='the size of a grid world')
    parser.add_argument('--n_actions', default=5, type=int, help='total number of actions an agent can take')
    parser.add_argument('--filename', default='../data/pr.txt', type=str, help='Pick-up probability file')
    parser.add_argument('--n_agents', default=1, type=int, help='the number of agent play in the environment')
    parser.add_argument('--runs', default=1, type=int, help='the number of times run the game')

    # parser args
    args = parser.parse_args()
    env = GridWorld(args=args, terminal_time=1000, reward_stay=-1, reward_hitwall=-2, reward_move=-1, reward_pick=2, aggre=False)

    # Create memory
    memory = ReplayMemory(buffer=50000, batchSize=500)

    # Create a network
    dqn = DQN(memory=memory)
    # Evaluating......
    print('\nCollecting experience...')
    for i_episode in range(4000):
        s, done = env.reset()
        ep_r = 0
        while True:
            (x, y) = s[0]['loc']
            one_hot_state = torch.Tensor(100 * 100)
            one_hot_state[x*args.grid_size+y] = 1
            a = torch.from_numpy(dqn.choose_action(one_hot_state))
            # take an action return next_state, reward, done, info
            torch.unsqueeze(a, 0)
            s_, r, done, info = env.step(a)
            # modify the reward
            # x, x_dot, theta, theta_dot = s_
            dqn.store_transition(s, a, r, s_, done)

            ep_r += r[0]
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done[0]:
                    print('Ep:', i_episode,
                          '| Ep_r:', round(ep_r, 2))

            if done[0]:
                break
            s = s_
