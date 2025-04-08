import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
import numpy as np

from DQNClasses.model import DQN, DQNCNN
import numpy as np

device = 'cpu'


class Agent:
    def __init__(self, discount, learning_rate, n_state, n_action, memory_size, muti_dim=1, IS_CNN=0):
        self.discount = discount
        self.learning_rate = learning_rate
        self.n_state = n_state
        self.n_action = n_action
        self.memory = ReplayMemory(n_state, memory_size)
        if IS_CNN == 1:
            self.action_network = DQNCNN(self.n_state, self.n_action).to(device)
            self.target_network = DQNCNN(self.n_state, self.n_action).to(device)
        else:
            self.action_network = DQN(self.n_state, self.n_action, muti_dim).to(device)
            self.target_network = DQN(self.n_state, self.n_action, muti_dim).to(device)
        self.target_network.load_state_dict(self.action_network.state_dict())
        # self.optimizer = torch.optim.RMSprop(params=self.action_network.parameters(), lr=learning_rate, momentum=0.95, eps=0.01)
        self.optimizer = torch.optim.Adam(params=self.action_network.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        self.criterion = torch.nn.MSELoss().to(device)
    # DDQN
    def update_training_network(self, batch_size, double_q=True):
        if self.memory.count < batch_size:
            return 0.0
        # 从经验回放数组中采样
        batch_state, batch_action, batch_reward, batch_next_state, batch_dones = self.memory.sample(batch_size)
        # 当前状态的action-state value
        batch_q_values = self.action_network(batch_state).gather(dim=1, index=batch_action)
        # training network计算下一状态的所有动作价值
        if double_q == True:
            batch_next_q_values = self.action_network(batch_next_state)
            batch_next_target_q_values = self.target_network(batch_next_state).detach()
            opt_action = torch.max(batch_next_q_values, 1)[1].unsqueeze(1)
            batch_target_values = batch_next_target_q_values.gather(1, opt_action)
        else:
            batch_target_values = self.target_network(batch_next_state).max(1)[0].unsqueeze(1).detach()
        
        

        batch_td_target = batch_reward + self.discount * batch_target_values * (1 - batch_dones)
        # batch_td_target = batch_reward + self.discount * batch_target_values

        loss = self.criterion(batch_q_values, batch_td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.action_network.state_dict())

    def choose_action(self, epis, state):
        rand_i = np.random.random()
        if rand_i < epis:
            predict_actions = np.random.randint(0, self.n_action)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).detach()
            Q_value = self.action_network(state)
            predict_actions = torch.argmax(Q_value).item()

        return predict_actions
    
    def choose_action_test(self, epis, state):
        # rand_i = np.random.random()
        # if rand_i < epis:
        #     predict_actions = np.random.randint(0, self.n_action)
        # else:
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).detach()
        Q_value = self.action_network(state)
        predict_actions = torch.argmax(Q_value).item()

        return predict_actions


class ReplayMemory:
    def __init__(self, n_state, memory_size):


        self.memory_size = memory_size
        self.n_state = n_state
        self.replay_action = np.empty(self.memory_size, dtype=np.int64)
        self.replay_state = np.empty((self.memory_size, self.n_state), dtype=np.float32)
        self.replay_state_next = np.empty((self.memory_size, self.n_state), dtype=np.float32)
        self.replay_reward = np.empty(self.memory_size, dtype=np.float32)
        self.replay_done = np.empty(self.memory_size, dtype=np.bool_)
        self.count = 0
        self.current = 0


    def push(self, state, action, reward, next_state, done):
        self.replay_state[self.current] = state
        self.replay_action[self.current] = action
        self.replay_reward[self.current] = reward
        self.replay_state_next[self.current] = next_state
        self.replay_done[self.current] = done
        self.current = (self.current + 1) % self.memory_size
        self.count = min(self.count + 1, self.memory_size)

    def sample(self, batch_size):
        if self.count < batch_size:
            indexes = random.sample(range(0, self.count), self.count)
        else:
            indexes = random.sample(range(0, self.count), batch_size)
        batch_state = self.replay_state[indexes]
        batch_action = self.replay_action[indexes]
        batch_reward = self.replay_reward[indexes]
        batch_next_state = self.replay_state_next[indexes]
        batch_dones = self.replay_done[indexes]
        
        
        batch_state = torch.tensor(batch_state, dtype=torch.float32, device=device)
        batch_action = torch.tensor(batch_action, dtype=torch.int64, device=device).unsqueeze(1)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=device).unsqueeze(1)
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32, device=device)
        batch_dones = torch.tensor(batch_dones, dtype=torch.float32, device=device).unsqueeze(1)

        return batch_state, batch_action, batch_reward, batch_next_state, batch_dones
    

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)
class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)
        
        

