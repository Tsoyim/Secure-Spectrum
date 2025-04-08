import numpy as np
import torch as T
import torch.autograd
import torch.nn.functional as F
from DDPGClasses.agent import Agent
from DDPGClasses.functions import *
class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agent, n_action_c, n_action_p, alpha, beta, gamma, tau, actor_mutli=1, critic_mutli=1, device='cpu'):
        self.agents = []
        self.n_agent = n_agent
        self.n_action_c = n_action_c
        self.n_action_p = n_action_p
        self.device = device
        for agent_idx in range(self.n_agent):
            self.agents.append(Agent(n_agent, actor_dims, critic_dims, alpha, beta, gamma, tau, n_action_c, n_action_p, actor_mutli, critic_mutli, device))



    def learn(self, memory):
        if not memory.ready():
            return
        # with torch.autograd.detect_anomaly():
        # 采样
        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.device

        states = T.tensor(states, dtype=T.float32).to(device)
        actions = T.tensor(actions, dtype=T.float32).to(device)
        rewards = T.tensor(rewards, dtype=T.float32).to(device)
        states_ = T.tensor(states_, dtype=T.float32).to(device)
        dones = T.tensor(dones, dtype=T.float32).to(device)

        all_agents_new_actions = []
        old_agents_actions = []
        all_agents_new_mu_actions = []
        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], dtype=T.float32).to(device)
            # target network不要梯度
            new_pi = agent.target_actor.forward(new_states)

            new_pi = T.cat([new_pi[0], new_pi[1]], dim=1).to(device).detach()

            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float32).to(device)

            mu = agent.actor.forward(mu_states)
            mu = T.cat([mu[0], mu[1]], dim=1).to(device).detach()
            all_agents_new_mu_actions.append(mu)
            all_agents_new_actions.append(new_pi)
            old_agents_actions.append(actions[agent_idx])
        # 通过target网络得到，用于计算Q'(s',a')
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        # 经验回放数组

        critic_loss_arr = []
        for agent_idx, agent in enumerate(self.agents):

            # agent.detect()
            old_agents_actions_tmp = old_agents_actions.copy()
            all_agents_new_mu_actions_tmp = all_agents_new_mu_actions.copy()
            # 获取trg crititc value


            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten().detach()



            # 获取critic value
            old_actions = T.cat([acts for acts in old_agents_actions_tmp], dim=1).clone()
            critic_value = agent.critic.forward(states, old_actions).flatten()
            # 通过actor网络获取当前actor网络取得的动作
            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float32).to(device)
            mu = agent.actor.forward(mu_states)
            mu = T.cat([mu[0], mu[1]], dim=1).to(device)

            old_agents_actions_tmp[agent_idx] = mu
            mu_actions = T.cat([acts for acts in old_agents_actions_tmp], dim=1)


            # 计算Critic网络的更新和Actor网络的更新
            target = rewards[:, agent_idx] + agent.gamma * critic_value_ * (1 - dones[:, agent_idx])
            critic_loss = F.mse_loss(target, critic_value)



            agent.opt_critic.zero_grad()
            critic_loss.backward()
            agent.opt_critic.step()
            # 获取当前智能体动作价值
            actor_val = agent.critic.forward(states, mu_actions).flatten()
            actor_loss = -T.mean(actor_val)
            agent.opt_actor.zero_grad()
            actor_loss.backward()
            agent.opt_actor.step()

            agent.update_network_parameters()
            critic_loss_arr.append(critic_loss.item())
        return critic_loss_arr