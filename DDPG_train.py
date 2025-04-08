import argparse
from typing import List
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from RLenvHighWay import RLHighWayEnvironment
from DDPGClasses.maddpg import MADDPG
from DDPGClasses.buffer import MultiAgentReplayBuffer
from util import Util
import os
# torch.autograd.set_detect_anomaly(True)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
PRE_TRAIN_MODEL = 0
parser = argparse.ArgumentParser(description='for training')
parser.add_argument('--actor_lr', '-alr', help='actor学习率，默认0.001', default=1e-5, type=float)
parser.add_argument('--critic_lr', '-clr', help='critic学习率，默认0.001', default=5e-5, type=float)
parser.add_argument('--gamma', '-ga', help='折扣因子', default=0.99, type=float)
parser.add_argument('--batch_size', '-bs', help='mini-batch大小', default=256, type=int)
parser.add_argument('--episode', '-e', help='训练回合数', default=3000, type=int)
parser.add_argument('--num_V2V', '-nV2V', help='合法车辆个数', default=4, type=int)
parser.add_argument('--num_V2I', '-nV2I', help='V2I车辆个数', default=4, type=int)
parser.add_argument('--num_Eve', '-nEve', help='窃听个数', default=1, type=int)
parser.add_argument('--Eveknow', '-ek', help='窃听者是否可被观测', default=1, type=int)
parser.add_argument('--device', '-dv', help='训练网络设备', default='cuda')
parser.add_argument('--tau', '-ta', help='tau软更新系数', default=0.001, type=float)
# bl = 1 Round-Robin, bl = 2 Random, bl = 0 withoutbaseline


args = parser.parse_args()
WithBaseline = args.WithBaseline
actor_lr = args.actor_lr
critic_lr = args.critic_lr
gamma = args.gamma
batch_size = args.batch_size
episode = args.episode
# k_epochs = args.k_epochs
n_V2V = args.num_V2V
n_V2I = args.num_V2I
n_Eve = args.num_Eve
Eveknow = args.Eveknow
device = args.device
tau = args.tau

device = args.device
if device == 'cuda':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# else:
#     device = torch.device('cpu')

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
    "demand": int((800) * 8 * 2),
    "speed": 100 / 3.6,
    "seed": 123,
    "Eveknow": Eveknow,
    "EveDist": 5,
    "POSSION": 1
}
# 神经网络参数设置
fc1_dim = 512
fc2_dim = 256
fc3_dim = 128


env = RLHighWayEnvironment(config_environment)
env.time_block_limit = 10

env.init_simulation()

n_state = len(env.get_state())
n_action_c = 1
n_action_p = 1
n_action = n_action_c + n_action_p
n_agent = env.n_V2V
actor_dims = n_state
critic_dims = n_agent * n_state
actor_muti = int(env.n_V2V / 4)
critic_muti = int((env.n_V2V / 4))


maddpg_agents = MADDPG(actor_dims, critic_dims, n_agent, n_action_c, n_action_p, actor_lr, critic_lr, gamma, tau, actor_muti, critic_muti, device)
memory = MultiAgentReplayBuffer(100000, critic_dims, actor_dims, n_action, n_agent, batch_size=batch_size)
baseline_arr = ['no', 'Round-Robin', 'Random']
util = Util(env.n_V2I, env.n_V2V, env.n_Eve, env.Eveknow, f'MADDPG_{baseline_arr[WithBaseline]}')

if PRE_TRAIN_MODEL == 1:
    for i in range(env.n_V2V):
        i_agent = i
        _, model_path = util.get_model_path('ep_1000_agent_actor_{}_V2I_{}_V2V_{}_Eve_{}.pt'.format(i_agent, env.n_V2I, env.n_V2V, env.n_Eve))
        maddpg_agents.agents[i].actor.load_state_dict(torch.load(model_path))
        maddpg_agents.agents[i].target_actor.load_state_dict(torch.load(model_path))
        # _, model_path = util.get_model_path('agent_critic_{}_V2I_{}_V2V_{}_Eve_{}.pt'.format(i_agent, env.n_V2I, env.n_V2V, env.n_Eve))
        # maddpg_agents.agents[i].critic.load_state_dict(torch.load(model_path))
        # maddpg_agents.agents[i].target_critic.load_state_dict(torch.load(model_path))

# 记录奖励
reward_record = []
var_start_pw = 0.8
var_end_pw = 0.01
var_start_ch = 0.5
var_end_ch = 0.01
# 初始化噪声
var_pw = var_start_pw
var_ch = var_start_ch

noise_decay_episode = int(episode * 0.8)
decay_per_episode_pw = (var_start_pw - var_end_pw) / (noise_decay_episode)
decay_per_episode_ch = (var_start_ch - var_end_ch) / (noise_decay_episode)
exploration = True
for i_ep in range(episode):
    print('-------episode: {} / {}  device: {}-------'.format(i_ep, episode - 1, device))
    env.update_env_slow(False)
    if (i_ep + 1) % 20 == 0:
        env.update_env_slow()

    action_ch = np.zeros([env.n_V2V], dtype=np.int64)
    action_pw = np.zeros([env.n_V2V], dtype=np.float32)
    # if i_ep < explor_episode:
    #     exploration = True
    # else:
    #     exploration = False
    reward_ep = 0
    for i_step in range(env.time_block_limit):
        obs_old_all = []
        obs_next_old_all = []
        vals_old_all = []
        action_all = []
        action_probs_all = []
        # time_step = i_step + i_ep * ep_step
        for i in range(env.n_V2V):
            i_agent = i
            indexes = i
            obs_n = env.get_state(indexes, ind_episode=i_ep / episode)
            obs_old_all.append(obs_n)
            ch, pw = maddpg_agents.agents[i].choose_action(obs_n, var_pw, var_ch)
            action = np.concatenate([ch, pw])
            action_all.append(action)
            ch = (np.clip(ch, -0.9999, 0.9999) + 1) / 2
            pw = (np.clip(pw, -1, 1) + 1) / 2
            a_pw = 23 * pw
           
            # a_pw = np.log10(pw*200 + 1e-10)
            a_ch = int(ch * env.n_V2I)


            action_ch[i] = a_ch
            action_pw[i] = a_pw
        # g_states = env.get_gstate()
        # 执行动作
        V2I_rate, V2V_rate, Secrecy_rate, reward = env.step_conti(action_ch.copy(), action_pw.copy())

        reward_ep += np.average(reward)
        if i_step == env.time_block_limit - 1:
            done = True
        else:
            done = False
        for i in range(env.n_V2V):
            indexes = i
            obs_next = env.get_state(indexes, ind_episode=i_ep / episode)
            obs_next_old_all.append(obs_next)
        # g_states_next = env.get_gstate()
        memory.store_transition(obs_old_all, action_all, reward, obs_next_old_all, done)

    critic_loss_arr = maddpg_agents.learn(memory)
    if critic_loss_arr != None:

        print('critic loss: {:.3f}'.format(np.mean(critic_loss_arr)))
    print('reward     : {:.3f}'.format(reward_ep))
    # 噪声衰减
    var_ch = var_ch - decay_per_episode_ch if var_ch - decay_per_episode_ch >= var_end_ch else var_end_ch
    var_pw = var_pw - decay_per_episode_pw if var_pw - decay_per_episode_pw >= var_end_pw else var_end_pw
    # 保存模型
    if (i_ep + 1) >= 1000 and (i_ep + 1) % 500 == 0:
        for i in range(env.n_V2V):
            i_agent = i
            _, path = util.get_model_path(
                'ep_{}_agent_actor_{}_V2I_{}_V2V_{}_Eve_{}.pt'.format(i_ep + 1, i_agent, env.n_V2I, env.n_V2V, env.n_Eve))
            torch.save(maddpg_agents.agents[i].actor.state_dict(), path)
    reward_record.append(reward_ep)

for i in range(env.n_V2V):
    i_agent = i
    _, path = util.get_model_path(
        'agent_actor_{}_V2I_{}_V2V_{}_Eve_{}.pt'.format(i_agent, env.n_V2I, env.n_V2V, env.n_Eve))
    torch.save(maddpg_agents.agents[i].actor.state_dict(), path)
    print(path)
    _, path = util.get_model_path(
        'agent_critic_{}_V2I_{}_V2V_{}_Eve_{}.pt'.format(i_agent, env.n_V2I, env.n_V2V, env.n_Eve))
    torch.save(maddpg_agents.agents[i].critic.state_dict(), path)
    print(path)
_, fig_path, reward_data_path = util.get_train_result_path('reward',
                                                           'reward_V2I_{}_V2V_{}_Eve_{}'.format(env.n_V2I, env.n_V2V,
                                                                                                env.n_Eve))
plt.plot(list(range(1, episode + 1)), reward_record, label='reward')
plt.legend()
plt.grid(True)
plt.savefig(fig_path)

# plt.show()
np.save(reward_data_path, arr=np.asarray(reward_record))







