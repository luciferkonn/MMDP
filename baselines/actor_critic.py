import argparse
import json

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Actor Network
# from utils.dotdic import DotDic
from grid_city import GridWorld
from torch.autograd import Variable

# Global Constants
STATE_DIM = 100 * 100
N_EPISODES = 4000
SAMPLE_NUMS = 30


# Define Actor Network
class ActorNet(nn.Module):
    def __init__(self, input_size, hidden_size, n_action):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_action)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        output = F.log_softmax(self.fc3(out)[0])
        return output


# Define Critic Network
class CriticNet(nn.Module):
    def __init__(self, n_state, hidden_size, output_size):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(n_state, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output


# state to one hot state
def state_to_oh(state):
    state_oh = np.zeros((1, STATE_DIM))
    (x, y) = state[0]['loc']
    state_oh[0, x*100+y] = 1
    return state_oh


def roll_out(actor_net, env, sample_nums, critic_net, init_state, args):
    states = []
    actions = []
    rewards = []
    is_done = False
    final_r = 0
    state = init_state

    for j in range(sample_nums):
        states.append(state)
        log_softmax_action = actor_net(Variable(torch.Tensor([state])))
        softmax_action = torch.exp(log_softmax_action)
        action = np.array([np.random.choice(args.n_actions, p=softmax_action.cpu().data.numpy()[0])]).astype(int)
        one_hot_action = [int(k == action) for k in range(args.n_actions)]
        next_state, reward, done, _ = env.step(action)
        actions.append(one_hot_action)
        rewards.append(reward)
        final_state = next_state
        state = next_state
        if done:
            is_done = True
            state, _ = env.reset()
            state = state_to_oh(state)
            break
    if not is_done:
        final_r = critic_net(Variable(torch.Tensor([final_state]))).cpu().data.numpy()

    return states, actions, rewards, final_r, state


def discount_reward(r, gamma, final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[0][t]
        discounted_r[t] = running_add
    return discounted_r


def main():
    parser = argparse.ArgumentParser(description='Multi-agent A2C')
    # add argument
    parser.add_argument('-c', "--config_path", type=str, help='the path of the config file')
    parser.add_argument('-r', '--result_path', default='../results/', type=str, help='the path of the results file')
    # will be replaced by config file
    parser.add_argument('--grid_size', default=100, type=int, help='the size of a grid world')
    parser.add_argument('--n_actions', default=7, type=int, help='total number of actions an agent can take')
    parser.add_argument('--filename', default='../data/pr.txt', type=str, help='Pick-up probability file')
    parser.add_argument('--n_agents', default=4, type=int, help='the number of agent play in the environment')
    parser.add_argument('--runs', default=1, type=int, help='the number of times run the game')
    parser.add_argument('--aggre', default=False, help='the number of times run the game')
    # parser args
    args = parser.parse_args()

    # opt = DotDic(json.loads(open(args.config_path, 'r').read()))
    ACTION_DIM = args.n_actions
    env = GridWorld(args=args, terminal_time=1000, reward_stay=-0.1, reward_hitwall=-1, reward_move=-0.1, reward_pick=10)
    init_state, done = env.reset()
    init_state = state_to_oh(init_state)

    # init value network
    critic_net = CriticNet(n_state=STATE_DIM, hidden_size=40, output_size=1)
    critic_net_optim = torch.optim.Adam(critic_net.parameters())

    # init actor network
    actor_net = ActorNet(input_size=STATE_DIM, hidden_size=40, n_action=ACTION_DIM)
    actor_net_optim = torch.optim.Adam(actor_net.parameters())

    steps = []
    task_episodes = []
    test_results = []

    for i_episode in range(N_EPISODES):
        states, actions, rewards, final_r, current_state = roll_out(actor_net, env, SAMPLE_NUMS, critic_net, init_state,
                                                                    args)
        init_state = current_state
        actions_var = Variable(torch.Tensor(actions).view(-1, ACTION_DIM))
        states_var = Variable(torch.Tensor(states).view(-1, STATE_DIM))

        # train actor network
        actor_net_optim.zero_grad()
        log_softmax_actions = actor_net(states_var)
        vs = critic_net(states_var).detach()

        # calculate qs
        qs = Variable(torch.Tensor(discount_reward(rewards, 0.99, final_r)))

        advantages = qs - vs
        actor_net_loss = -torch.mean(torch.sum(log_softmax_actions*actions_var)*advantages)
        actor_net_loss.backward()
        nn.utils.clip_grad_norm(actor_net.parameters(), 0.5)
        actor_net_optim.step()

        # train critic network
        critic_net_optim.zero_grad()
        target_values = qs
        values = critic_net(states_var)
        criterion = nn.MSELoss()
        critic_net_loss = criterion(values, target_values)
        critic_net_loss.backward()
        nn.utils.clip_grad_norm(critic_net.parameters(), 0.5)
        critic_net_optim.step()

        # Testing
        # if (i_episode + 1) % 50 == 0:
        # for test_epi in range(4000):
        state, _ = env.reset()
        state = state_to_oh(state)
        result = 0
        while True:
            softmax_action = torch.exp(actor_net(Variable(torch.Tensor([state]))))
            # print(softmax_action.data)
            action = np.array([np.argmax(softmax_action.data.numpy()[0])]).astype(int)
            next_state, reward, done, _ = env.step(action)
            next_state = state_to_oh(next_state)
            result += reward[0]
            state = next_state
            if done[0]:
                print("step:", i_episode + 1, "test result:", round(result, 2))
                steps.append(i_episode + 1)
                test_results.append(result / 10)
                break


if __name__ == '__main__':
    main()
