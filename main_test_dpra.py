import argparse
import torch
import numpy as np
from typing import List
import os
from DDPGClasses.maddpg import MADDPG
from torch.distributions import Categorical
from RLenvHighWay_test import RLHighWayEnvironment
from util import Util
from DQNClasses.agent import Agent
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from exhaustive import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cpu = 'cpu'
parser = argparse.ArgumentParser(description='for test')
parser.add_argument('--test_episode', '-te', help='测试回合数', default=200, type=int)
parser.add_argument('--num_V2V', '-nV2V', help='合法车辆个数', default=6, type=int)
parser.add_argument('--num_V2I', '-nV2I', help='V2I车辆个数', default=6, type=int)
parser.add_argument('--num_Eve', '-nEve', help='窃听车辆个数', default=1, type=int)
parser.add_argument('--Eveknow', '-ke', help='窃听者是否可被观测', default=1, type=int)
parser.add_argument('--Speed', '-s', help='车辆移动速度', default=100, type=float)
parser.add_argument('--eve_dist', '-ed', help='窃听者距离', default=5, type=int)
parser.add_argument('--multiDemand', '-mD', help='数据包大小', default=4.0, type=float)
# bl = 1 Round-Robin, bl = 2 Random, bl = 0 withoutbaseline
parser.add_argument('--WithBaseline', '-bl', help='奖励函数的baseline', default=0, type=int)
parser.add_argument('--dqn_ep', '-qe', help='..', default=3000, type=int)
parser.add_argument('--ddpg_ep', '-ge', help='..', default=3000, type=int)

args = parser.parse_args()
WithBaseline = args.WithBaseline
test_episode = args.test_episode
# dir_name = args.dir_name
Eveknow = args.Eveknow
n_V2V = args.num_V2V
n_V2I = args.num_V2I
n_Eve = args.num_Eve
speed = args.Speed
EveDist = args.eve_dist
dqn_ep = args.dqn_ep
ddpg_ep = args.ddpg_ep
mD = args.multiDemand
config_environment = {
    "powerV2VdB": [23, 15, 5, -100],
    "powerV2I": 23,
    "backgroundNoisedB": -114,
    "n_V2V": n_V2V,
    "n_V2I": n_V2I,
    "n_Eve": n_Eve,
    "time_fast_fading": 0.001,
    "time_slow_fading": 0.1,
    "bandwidth": int(1e6),
    "demand": int((800) * 8 * mD),
    "speed": speed / 3.6,
    "seed": 123,
    "Eveknow": Eveknow,
    "POSSION": 0,
    "EveDist": EveDist,
    "3GPP": 37
}
# IS_HIGHWAY = args.is_highway
env = RLHighWayEnvironment(config_environment)
env.time_block_limit = 10
env.init_simulation()
n_state = len(env.get_state_dqn())
n_action = len(env.power_list_V2V_dB) * env.n_V2I
n_action_c = 1
n_action_p = 1
n_action_ddpg = 2
n_agent = env.n_V2V
actor_dims = n_state
critic_dims = n_agent * n_state
# 网络维度
actor_muti = int(env.n_V2V / 4)
# actor_muti = int(np.ceil(env.n_V2V / 4))
critic_muti = int(env.n_V2V / 4)
# DQN网络维度
Q_muti = int(env.n_V2V / 4)


agent_list: List[Agent] = []
baseline_arr = ['no', 'Round-Robin', 'Random']
util_dqn = Util(env.n_V2I, env.n_V2V, env.n_Eve, env.Eveknow, f'MADQN_{baseline_arr[WithBaseline]}')
util_ddpg = Util(env.n_V2I, env.n_V2V, env.n_Eve, env.Eveknow, f'MADDPG_{baseline_arr[WithBaseline]}')
for i in range(env.n_V2V):
    _, model_path = util_dqn.get_model_path(
        'ep_{}_agent_{}_V2I_{}_V2V_{}_Eve_{}.pt'.format(dqn_ep,i, env.n_V2I, env.n_V2V, env.n_Eve))
    print(model_path)
    current_agent = Agent(0.99, 0.01, n_state, n_action, 10, muti_dim=Q_muti)
    agent_list.append(current_agent)
    current_agent.action_network.load_state_dict(torch.load(model_path))

maddpg_agents = MADDPG(actor_dims, critic_dims, n_agent, n_action_c, n_action_p, 0.1, 0.1, 0.1, 0.1, device='cuda', actor_mutli=actor_muti, critic_mutli=critic_muti)

for i in range(env.n_V2V):
    i_agent = i
    _, model_path = util_ddpg.get_model_path('ep_{}_agent_actor_{}_V2I_{}_V2V_{}_Eve_{}.pt'.format(ddpg_ep, i_agent, env.n_V2I, env.n_V2V, env.n_Eve))
    maddpg_agents.agents[i].actor.load_state_dict(torch.load(model_path))

# env.plot_dynamic_car()
n_state = len(env.get_state_dqn())
n_action = len(env.power_list_V2V_dB) * env.n_V2I

sum_V2I_dpra = []
sum_Sec_dpra = []
SecTrans_Pro_dpra = []

sum_V2I_dqn = []
sum_Sec_dqn = []
SecTrans_Pro_dqn = []

sum_V2I_ddpg = []
sum_Sec_ddpg = []
SecTrans_Pro_ddpg = []

sum_V2I_rand = []
sum_Sec_rand = []
SecTrans_Pro_rand = []

time_step = 0
max_V2I = []

# env.init_simulation()
for i_ep in range(test_episode):
    print('-------testepisode: {} / {}  device: {}-------'.format(i_ep, test_episode - 1, device))

    env.update_env_slow()
    # -------DPRA-----------
    action_test_dpra = np.zeros([env.n_V2V, 2], dtype='int64')
    # -------MADQN-----------
    action_all_dqn = np.zeros([env.n_V2V, 2], dtype='int')
    # -------Rand-----------
    action_all_rand = np.zeros([env.n_V2V, 2], dtype='int')
    # -------MADDPG-----------
    action_ddpg_ch = np.zeros([env.n_V2V], dtype='int')
    action_ddpg_pw = np.zeros([env.n_V2V], dtype='float')
    reward_ep = 0

    for i_step in range(env.time_block_limit):

        # ---------------dpra-------------------------------
        n_power_level = 1
        store_action = np.zeros([(env.n_Channel * n_power_level) ** 4, 4])
        rate_all_dpra = []
        t = 0
        # for i in range(n_RB*len(env.V2V_power_dB_List)):\
        if env.n_V2V == 4:
            action_test_dpra = search4(env)
        if env.n_V2V == 6:
            action_test_dpra = search6(env)
        if env.n_V2V == 8:
            action_test_dpra = search8(env)
        if env.n_V2V == 10:
            action_test_dpra = search10(env)


        # --------------------DRL-----------------------------------------------
        for i in range(env.n_V2V):
            # ----------dqn--------------
            current_agent = agent_list[i]
            indexes = i
            state = env.get_state_dqn(indexes)
            action_dqn = current_agent.choose_action(0, state)
            # channel action
            action_all_dqn[i, 0] = action_dqn / (len(env.power_list_V2V_dB))
            action_all_dqn[i, 1] = action_dqn % (len(env.power_list_V2V_dB))

            # ----------ddpg--------------
            obs_n = env.get_state_ddpg(i)
            ch, pw = maddpg_agents.agents[i].choose_action_test(obs_n)
            ch = (np.clip(ch, -0.9999, 0.9999) + 1) / 2
            pw = (np.clip(pw, -1, 1) + 1) / 2
            pw = 24 * pw - 1
            if pw >= 0:
                a_pw = pw
            else:
                a_pw = -100
            a_ch = int(ch * env.n_V2I)
            action_ddpg_ch[i] = a_ch
            action_ddpg_pw[i] = a_pw

        # ----------random--------------
        action_all_rand[:, 0] = np.random.randint(0, env.n_Channel, env.n_V2V)
        action_all_rand[:, 1] = np.random.randint(0, env.n_power_level, env.n_V2V)
        # 各个算法执行动作
        V2I_rate_dpra, V2V_rate_dpra, Secrecy_rate_dpra = env.step_dpra(action_test_dpra.copy())
        V2I_rate_rand, V2V_rate_rand, Secrecy_rate_rand = env.step_rand(action_all_rand.copy())
        V2I_rate_dqn, V2V_rate_dqn, Secrecy_rate_dqn = env.step_dqn(action_all_dqn.copy())
        V2I_rate_ddpg, V2V_rate_ddpg, Secrecy_rate_ddpg = env.step_ddpg(action_ddpg_ch.copy(), action_ddpg_pw.copy())

        max_V2I.append(np.sum(env.compute_max_V2I()))

        # 更新小尺度衰落
        env.update_env_fast()
        env.compute_V2V_interference(action_all_dqn.copy(), mode='dqn')
        env.compute_V2V_interference_ddpg(action_ddpg_ch.copy(), action_ddpg_pw.copy())

        # 记录数据
        sum_Sec_dpra.append(np.sum(Secrecy_rate_dpra))
        sum_V2I_dpra.append(np.sum(V2I_rate_dpra))

        sum_Sec_dqn.append(np.sum(Secrecy_rate_dqn))
        sum_V2I_dqn.append(np.sum(V2I_rate_dqn))

        sum_Sec_ddpg.append(np.sum(Secrecy_rate_ddpg))
        sum_V2I_ddpg.append(np.sum(V2I_rate_ddpg))

        sum_Sec_rand.append(np.sum(Secrecy_rate_rand))
        sum_V2I_rand.append(np.sum(V2I_rate_rand))

    SecTrans_Pro_dpra.append(np.sum(env.active_links_dpra == 0) / env.n_V2V)
    SecTrans_Pro_ddpg.append(np.sum(env.active_links_ddpg == 0) / env.n_V2V)
    SecTrans_Pro_dqn.append(np.sum(env.active_links_dqn == 0) / env.n_V2V)
    SecTrans_Pro_rand.append(np.sum(env.active_links_rand == 0) / env.n_V2V)
    print('         Random                     DDPG        Exhaustive')
    print('V2I Rate: {:.3f}                 {:.3f}        {:.3f}'.format(np.average(np.asarray(sum_V2I_rand)),
                                                                                np.average(np.asarray(sum_V2I_ddpg)),
                                                                                np.average(np.asarray(sum_V2I_dpra))))
    print(
        'SecTrans: {:.3f}                {:.3f}        {:.3f}'.format(np.average(np.asarray(SecTrans_Pro_rand)),
                                                                              np.average(np.asarray(SecTrans_Pro_ddpg)),
                                                                              np.average(
                                                                                  np.asarray(SecTrans_Pro_dpra))))
    print('Sec Rate: {:.3f}                  {:.3f}        {:.3f}'.format(np.average(np.asarray(sum_Sec_rand)),
                                                                                np.average(np.asarray(sum_Sec_ddpg)),
                                                                                np.average(np.asarray(sum_Sec_dpra))))

    print('Max V2I:{:.4f}'.format(np.average(np.asarray(max_V2I))))

data_path = f"./Data/V2I_{env.n_V2I}_V2V_{env.n_V2V}_Eve_{env.n_Eve}/数据/V2I_{env.n_V2I}_V2V_{env.n_V2V}_Eve_{env.n_Eve}_size{mD}_ed{EveDist}.csv"

with open(data_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['', 'Random', 'DQN', 'DDPG', 'DPRA'])
    writer.writerow(['V2I Rate', '{:.3f}'.format(np.average(np.asarray(sum_V2I_rand))),
                     '{:.3f}'.format(np.average(np.asarray(sum_V2I_dqn))),
                     '{:.3f}'.format(np.average(np.asarray(sum_V2I_ddpg))),
                     '{:.3f}'.format(np.average(np.asarray(sum_V2I_dpra)))])  # Add DPRA column
    writer.writerow(['SecTrans', '{:.3f}'.format(np.average(np.asarray(SecTrans_Pro_rand))),
                     '{:.3f}'.format(np.average(np.asarray(SecTrans_Pro_dqn))),
                     '{:.3f}'.format(np.average(np.asarray(SecTrans_Pro_ddpg))),
                     '{:.3f}'.format(np.average(np.asarray(SecTrans_Pro_dpra)))])  # Add DPRA column
    writer.writerow(['Sec Rate', '{:.3f}'.format(np.average(np.asarray(sum_Sec_rand))),
                     '{:.3f}'.format(np.average(np.asarray(sum_Sec_dqn))),
                     '{:.3f}'.format(np.average(np.asarray(sum_Sec_ddpg))),
                     '{:.3f}'.format(np.average(np.asarray(sum_Sec_dpra)))])  # Add DPRA column

