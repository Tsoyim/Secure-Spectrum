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
import time
import matplotlib.pyplot as plt
import seaborn as sns
import csv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
parser = argparse.ArgumentParser(description='for test')
parser.add_argument('--agent', '-n', help='智能体数量', default=8, type=int)
parser.add_argument('--eve', '-e', help='窃听者数量', default=1, type=int)
parser.add_argument('--speed', '-s', help='速度', default=100.0, type=float)
parser.add_argument('--speed_diff', '-sd', help='速度', default=20.0, type=float)
parser.add_argument('--multiDemand', '-mD', help='数据包大小', default=4.0, type=float)
# bl = 1 Round-Robin, bl = 2 Random, bl = 0 withoutbaseline


args = parser.parse_args()
WithBaseline = 0
test_episode = 200
# dir_name = args.dir_name
Eveknow = 1
n_V2V = args.agent
n_V2I = n_V2V
n_Eve = args.eve
dqn_ep = 3000
ddpg_ep = 3000
speed = args.speed
EveDist = 0
speed_diff = args.speed_diff
mD = args.multiDemand
config_environment = {
    "powerV2VdB": [23, 15, 5, 0],
    "powerV2I": 23,
    "backgroundNoisedB": -114,
    "n_V2V": n_V2V,
    "n_V2I": n_V2I,
    "n_Eve": n_Eve,
    "time_fast_fading": 0.001,
    "time_slow_fading": 0.1,
    "bandwidth": int(1e6),
    "demand": int((800) * 8 * mD),
    "speed": speed,
    "speed_diff": speed_diff,
    "seed": 123,
    "Eveknow": Eveknow,
    "POSSION": 1,
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
# -------------DDPG网络维度--------------
actor_dims = n_state
critic_dims = n_agent * n_state
actor_muti = int(env.n_V2V / 4)
# actor_muti = int(np.ceil(env.n_V2V / 4))
critic_muti = int(env.n_V2V / 4)
# DQN网络维度
Q_muti = int(env.n_V2V / 4)
# Q_muti = int(np.ceil(env.n_V2V / 4))
agent_list: List[Agent] = []

util_dqn = Util(env.n_V2I, env.n_V2V, env.n_Eve, env.Eveknow, f'MADQN_no')
util_ddpg = Util(env.n_V2I, env.n_V2V, env.n_Eve, env.Eveknow, f'MADDPG_no')
for i in range(env.n_V2V):

    _, model_path = util_dqn.get_model_path('ep_{}_agent_{}_V2I_{}_V2V_{}_Eve_{}.pt'.format(dqn_ep, i, env.n_V2I, env.n_V2V, 1))
    print(model_path)
    current_agent = Agent(0.99, 0.01, n_state, n_action, 10, muti_dim=Q_muti)
    agent_list.append(current_agent)
    current_agent.action_network.load_state_dict(torch.load(model_path))

maddpg_agents = MADDPG(actor_dims, critic_dims, n_agent, n_action_c, n_action_p, 0.1, 0.1, 0.1, 0.1, device=device, actor_mutli=actor_muti, critic_mutli=critic_muti)
for i in range(env.n_V2V):
    i_agent = i
    _, model_path = util_ddpg.get_model_path('ep_{}_agent_actor_{}_V2I_{}_V2V_{}_Eve_{}.pt'.format(ddpg_ep, i_agent, env.n_V2I, env.n_V2V, 1))
    maddpg_agents.agents[i].actor.load_state_dict(torch.load(model_path))


# env.plot_dynamic_car()
n_state = len(env.get_state_dqn())
n_action = len(env.power_list_V2V_dB) * env.n_V2I




sum_V2I_dqn = []
sum_Sec_dqn = []
SecTrans_Pro_dqn = []

sum_V2I_ddpg = []
sum_Sec_ddpg = []
SecTrans_Pro_ddpg = []

sum_V2I_rand = []
sum_Sec_rand = []
SecTrans_Pro_rand = []

sum_V2I_dpra = []
sum_Sec_dpra = []
SecTrans_Pro_dpra = []

all_dqn_time = []

all_ddpg_time = []

all_random_time = []

all_graph_time = []

time_step = 0
max_V2I = []

for i_ep in range(test_episode):
    print('-------testepisode: {} / {}  device: {}-------'.format(i_ep, test_episode - 1, device))
    env.update_env_slow(True)
    # -------DPRA-----------
    action_dpra = np.zeros([env.n_V2V, 2], dtype='int64')
    # -------MADQN-----------
    action_all_dqn = np.zeros([env.n_V2V, 2], dtype='int')
    # -------Rand-----------
    action_all_rand = np.zeros([env.n_V2V, 2], dtype='int')
    # -------MADDPG-----------
    action_ddpg_ch = np.zeros([env.n_V2V], dtype='int')
    action_ddpg_pw = np.zeros([env.n_V2V], dtype='float')
    reward_ep = 0


    for i_step in range(env.time_block_limit):

        # --------------------DRL-----------------------------------------------
        for i in range(env.n_V2V):

            # ----------dqn--------------
            dqn_time = time.time()
            current_agent = agent_list[i]
            indexes = i
            state = env.get_state_dqn(indexes)
            action_dqn = current_agent.choose_action_test(0.0, state)

            # channel action
            action_all_dqn[i, 0] = action_dqn / (len(env.power_list_V2V_dB))
            action_all_dqn[i, 1] = action_dqn % (len(env.power_list_V2V_dB))

            dqn_time = time.time() - dqn_time
            all_dqn_time.append(dqn_time)

            # ----------ddpg--------------
            ddpg_time = time.time()
            obs_n = env.get_state_ddpg(i)
            ch, pw = maddpg_agents.agents[i].choose_action_test(obs_n)
            # action_all.append(action)
            ch = (np.clip(ch, -0.9999, 0.9999) + 1) / 2
            pw = (np.clip(pw, -1, 1) + 1) / 2
            a_pw = 23 * pw
            
            # a_pw = np.log10(pw * 200 + 1e-10)
            a_ch = int(ch * env.n_V2I)
            # a_ch = int(ch * env.n_V2I)

            action_ddpg_ch[i] = a_ch
            action_ddpg_pw[i] = a_pw

            ddpg_time = time.time() - ddpg_time
            all_ddpg_time.append(ddpg_time)

            # ----------random--------------
            random_time = time.time()
            action_all_rand[i, 1] = np.random.randint(0, env.n_power_level)
            action_all_rand[i, 0] = np.random.randint(0, env.n_Channel)
            
            
            random_time = time.time() - random_time
            all_random_time.append(random_time)
        # 各个算法执行动作
        V2I_rate_rand, V2V_rate_rand, Secrecy_rate_rand = env.step_rand(action_all_rand.copy())
        V2I_rate_dqn, V2V_rate_dqn, Secrecy_rate_dqn = env.step_dqn(action_all_dqn.copy())
        V2I_rate_ddpg, V2V_rate_ddpg, Secrecy_rate_ddpg = env.step_ddpg(action_ddpg_ch.copy(), action_ddpg_pw.copy())
        graph_time = time.time()
        V2I_rate_dpra, V2V_rate_dpra, Secrecy_rate_dpra = env.step_IHA()
        graph_time = time.time() - graph_time
        all_graph_time.append(graph_time)
        max_V2I.append(np.sum(env.compute_max_V2I()))
        
        
        

        # 更新小尺度衰落
        env.update_env_fast()
        env.compute_V2V_interference_dqn(action_all_dqn.copy())
        env.compute_V2V_interference_ddpg(action_ddpg_ch.copy(), action_ddpg_pw.copy())


        # 记录数据
        sum_Sec_dqn.append(np.sum(Secrecy_rate_dqn))
        sum_V2I_dqn.append(np.sum(V2I_rate_dqn))

        sum_Sec_ddpg.append(np.sum(Secrecy_rate_ddpg))
        sum_V2I_ddpg.append(np.sum(V2I_rate_ddpg))

        sum_Sec_rand.append(np.sum(Secrecy_rate_rand))
        sum_V2I_rand.append(np.sum(V2I_rate_rand))
        
        
        sum_Sec_dpra.append(np.sum(Secrecy_rate_dpra))
        sum_V2I_dpra.append(np.sum(V2I_rate_dpra))

    SecTrans_Pro_ddpg.append(np.sum(env.active_links_ddpg == 0) / env.n_V2V)
    SecTrans_Pro_dqn.append(np.sum(env.active_links_dqn == 0) / env.n_V2V)
    SecTrans_Pro_rand.append(np.sum(env.active_links_rand == 0) / env.n_V2V)
    SecTrans_Pro_dpra.append(np.sum(env.active_links_dpra == 0) / env.n_V2V)
    print('         Random            DQN         DDPG           DPRA')
    print(
        'V2I Rate: {:.3f}               {:.3f}        {:.3f}         {:.3f}'.format(np.average(np.asarray(sum_V2I_rand)),
                                                                       np.average(np.asarray(sum_V2I_dqn)),
                                                                       np.average(np.asarray(sum_V2I_ddpg)),
                                                                       np.average(np.asarray(sum_V2I_dpra))))
    print('SecTrans: {:.3f}               {:.3f}        {:.3f}       {:.3f}'.format(np.average(np.asarray(SecTrans_Pro_rand)),
                                                                        np.average(np.asarray(SecTrans_Pro_dqn)),
                                                                        np.average(np.asarray(SecTrans_Pro_ddpg)),
                                                                        np.average(np.asarray(SecTrans_Pro_dpra))))
    # print(
    #     'Sec Rate: {:.3f}               {:.3f}        {:.3f}'.format(np.average(np.asarray(sum_Sec_rand)),
    #                                                                    np.average(np.asarray(sum_Sec_dqn)),
    #                                                                    np.average(np.asarray(sum_Sec_ddpg))))

    print('Max V2I:{:.3f}'.format(np.average(np.asarray(max_V2I))))
data_path = f"./Data/data/V2I_{env.n_V2I}_V2V_{env.n_V2V}_Eve_{env.n_Eve}_size{mD}_ed{EveDist}_s{env.speed*3.6}.csv"
directory = os.path.dirname(data_path)
if not os.path.exists(directory):
    os.makedirs(directory)

print('ddpg    平均每个时间戳耗时：{}'.format(np.average(np.asarray(all_ddpg_time))))
print('dqn     平均每个时间戳耗时：{}'.format(np.average(np.asarray(all_dqn_time))))
print('random  平均每个时间戳耗时：{}'.format(np.average(np.asarray(all_random_time)) / n_V2V))
# print('graph   平均每个时间戳耗时：{}'.format(np.average(graph_time)))
with open(data_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['', 'Random', 'DQN', 'DDPG', 'DPRA'])
    writer.writerow(['V2I Rate', '{:.3f}'.format(np.average(np.asarray(sum_V2I_rand))),
                     '{:.3f}'.format(np.average(np.asarray(sum_V2I_dqn))),
                     '{:.3f}'.format(np.average(np.asarray(sum_V2I_ddpg))),
                     '{:.3f}'.format(np.average(np.asarray(sum_V2I_dpra)))])
    writer.writerow(['SecTrans', '{:.3f}'.format(np.average(np.asarray(SecTrans_Pro_rand))),
                     '{:.3f}'.format(np.average(np.asarray(SecTrans_Pro_dqn))),
                     '{:.3f}'.format(np.average(np.asarray(SecTrans_Pro_ddpg))),
                     '{:.3f}'.format(np.average(np.asarray(SecTrans_Pro_dpra)))])
    # writer.writerow(['Sec Rate', '{:.3f}'.format(np.average(np.asarray(sum_Sec_rand))),
    #                  '{:.3f}'.format(np.average(np.asarray(sum_Sec_dqn))),
    #                  '{:.3f}'.format(np.average(np.asarray(sum_Sec_ddpg)))])
