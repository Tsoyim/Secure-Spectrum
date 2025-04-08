import random

import numpy as np
import torch as T
import torch.optim
import torch
from DDPGClasses.networks import ActorNetwork, CriticNetwork
from DDPGClasses.noise import OUNoise
import torch.nn.functional as F
from DDPGClasses.functions import *
class Agent:
    def __init__(self,n_agent,  actor_dims, critic_dims, alpha, beta, gamma, tau, n_action_c, n_action_p=1, actor_mutli=1, critic_mutli=1, device='cpu'):
        self.gamma = gamma
        self.tau = tau
        n_action = n_action_c + n_action_p
        self.n_action = n_action
        self.n_action_c = n_action_c
        self.n_action_p = n_action_p
        self.device = device
        self.actor = ActorNetwork(actor_dims, n_action_c, n_action_p, actor_mutli).to(device)
        self.critic = CriticNetwork(critic_dims, n_agent * (n_action), critic_mutli).net.to(device)
        self.target_actor = ActorNetwork(actor_dims, n_action_c, n_action_p, actor_mutli).to(device)
        self.target_critic = CriticNetwork(critic_dims, n_agent * (n_action), critic_mutli).net.to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        # self.target_actor.eval()
        # self.target_critic.eval()
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=alpha, weight_decay=1e-5)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=beta, weight_decay=1e-5)
        self.noise_ch = OUNoise()
        self.noise_pw = OUNoise()
        self.update_network_parameters(tau=tau)

    def choose_action(self, observation, var_pw, var_ch):
        # if exploration == True:
        #     ch = np.random.random(1) * 2 - 1
        #     pw = np.random.random(1) * 2 - 1
        #     return ch, pw
        # torch.unsqueeze()

        state = T.tensor(observation, dtype=T.float32).unsqueeze(0).to(self.device)
        ch, pw = self.actor.forward(state, True)


        noise1 = T.randn(1).to(self.device) * var_pw
        noise2 = T.randn(1).to(self.device) * var_ch
        # noise1 = T.tensor(self.noise_pw.noise()).to(self.device)
        # noise2 = T.tensor(self.noise_ch.noise()).to(self.device)
        # 给功率加噪声
        pw = pw + noise1
        ch = ch + noise2
        ch = ch[0].detach().cpu().numpy()
        pw = pw[0].detach().cpu().numpy()
        pw = np.clip(pw, -1, 1)
        ch = np.clip(ch, -1, 1)
        return ch, pw

    @torch.no_grad()
    def choose_action_test(self, observation):
        state = T.tensor([observation], dtype=T.float32).to(self.device)
        ch, pw = self.actor.forward(state, False)
        # noise1 = T.rand(self.n_action)
        return ch[0].detach().cpu().numpy(), pw[0].detach().cpu().numpy()


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        # 对参数进行软更新
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) +
                param.data * self.tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) +
                param.data * self.tau
            )




        # self.target_critic.load_state_dict(critic_state_dict)

    def detect(self):
        for param in self.actor.parameters():
            if torch.isnan(param).any():
                print("actor contains NaN values!")
        for param in self.critic.parameters():
            if torch.isnan(param).any():
                print("critic contains NaN values!")
        for param in self.target_actor.parameters():
            if torch.isnan(param).any():
                print("target actor contains NaN values!")
        for param in self.target_critic.parameters():
            if torch.isnan(param).any():
                print("target critic contains NaN values!")