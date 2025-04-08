import argparse
import random
from typing import List
import torch
import numpy as np
import matplotlib.pyplot as plt
from RLenvHighWay import RLHighWayEnvironment
from DQNClasses.agent import Agent
from util import Util
import os


# torch.autograd.detect_anomaly(True)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(1234)

parser = argparse.ArgumentParser(description='for training')
parser.add_argument('--learning_rate', '-lr', help='学习率，默认0.001', default=0.0001, type=float)
parser.add_argument('--discount', '-d', help='折扣因子', default=0.9, type=float)
parser.add_argument('--batch_size', '-b', help='mini-batch大小', default=256, type=int)
parser.add_argument('--memory_size', '-m', help='经验池大小', default=100000, type=int)
parser.add_argument('--episode', '-e', help='训练回合数', default=3000, type=int)
parser.add_argument('--epis_start', '-st', help='探索率', default=1, type=float)
parser.add_argument('--epis_end', '-en', help='探索率', default=0.02, type=float)
parser.add_argument('--anneal_rate', '-ar', help='退火率', default=0.8, type=float)
parser.add_argument('--update_ep', '-ue',help='更新目标网络的回合', default=4, type=int)
parser.add_argument('--num_V2V', '-nV2V', help='合法车辆个数', default=4, type=int)
parser.add_argument('--num_V2I', '-nV2I', help='V2I车辆个数', default=4, type=int)
parser.add_argument('--num_Eve', '-nEve', help='窃听车辆个数', default=1, type=int)
parser.add_argument('--Eveknow', '-ke', help='窃听者是否可被观测', default=1, type=int)
# bl = 1 Round-Robin, bl = 2 Random, bl = 0 withoutbaseline
parser.add_argument('--WithBaseline', '-bl', help='奖励函数的baseline', default=0, type=int)
parser.add_argument('--is_CNN', '-cnn', help='是否为cnn', default=0, type=int)
args = parser.parse_args()
WithBaseline = args.WithBaseline
learning_rate = args.learning_rate
discount = args.discount
batch_size = args.batch_size
memory_size = args.memory_size
episode = args.episode
epis_start = args.epis_start
epis_end = args.epis_end
anneal_rate = args.anneal_rate
update_ep = args.update_ep
Eveknow = args.Eveknow
n_V2V = args.num_V2V
n_V2I = args.num_V2I
n_Eve = args.num_Eve
IS_CNN = args.is_CNN
anneal_length = int(episode * anneal_rate)
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
    "speed": 100/3.6,
    "seed":123,
    "Eveknow":Eveknow,
    "EveDist":5,
    "POSSION":1,
    "3GPP":37
}


env = RLHighWayEnvironment(config_environment)
env.time_block_limit = 10
env.init_simulation()
# env.plot_dynamic_car()
n_state = len(env.get_state())
n_action = len(env.power_list_V2V_dB) * env.n_V2I
agent_list: List[Agent] = []
muti_dim = int(env.n_V2V / 4)
for i in range(env.n_V2V):
    agent_list.append(Agent(discount, learning_rate, n_state, n_action, memory_size, muti_dim, IS_CNN))
baseline_arr = ['no', 'Round-Robin', 'Random']
netlabels = ['no', 'cnn']
util = Util(env.n_V2I, env.n_V2V, env.n_Eve, env.Eveknow, f'MADQN_{netlabels[IS_CNN]}')





reward_record = []
ma_reward = []
loss_record = []
for i_ep in range(episode):
    print('-------episode: {} / {}  device: {}-------'.format(i_ep, episode - 1, device))
    env.update_env_slow(False)
    if (i_ep + 1) % 20 == 0:
        env.update_env_slow()
    if i_ep < anneal_length:
        epis = i_ep * (epis_start - epis_end) / (1 - anneal_length) + epis_start
    else:
        epis = epis_end

    # debug

    action_all_training = np.zeros([env.n_V2V, 2], dtype='int')
    action_all_baseline = np.zeros_like(action_all_training)
    reward_ep = 0
    for i_step in range(env.time_block_limit):
        state_old_all = []
        action_all = []
        for i in range(env.n_V2V):
            current_agent = agent_list[i]
            indexes = i
            state = env.get_state(indexes, epis, i_ep / episode)
            action = current_agent.choose_action(epis, state)
            state_old_all.append(state)
            action_all.append(action)
            # channel action
            action_all_training[i, 0] = action / (len(env.power_list_V2V_dB))
            action_all_training[i, 1] = action % (len(env.power_list_V2V_dB))

        _, _, _, reward = env.step(action_all_training.copy())
        reward = reward + np.zeros(env.n_V2V)
        reward_ep += np.average(reward)

        if i_step == env.time_block_limit - 1:
            done = True
        else:
            done = False
        # store
        env.update_env_fast()
        env.compute_V2V_interference(action_all_training.copy())
        for i in range(env.n_V2V):
            
            reward_i = reward[i]
            current_agent = agent_list[i]
            next_state = env.get_state(i, epis, i_ep / episode)
            current_agent.memory.push(state_old_all[i], action_all[i], reward_i, next_state, done)
    loss = 0
    for i in range(env.n_V2V):
        current_agent = agent_list[i]
        loss = current_agent.update_training_network(batch_size)
        if i_ep % (update_ep - 1) == 0:
            current_agent.update_target_network()
            if i == 0:
                print('update target network')
    if (i_ep + 1) >= 2000 and (i_ep + 1) % 500 == 0:
        for i in range(env.n_V2V):
            current_agent = agent_list[i]
            #  + 'V2I_{}_V2V_{}_Eve_{}'.format(n_CUE, n_VUE, n_Eve)
            _, path = util.get_model_path('ep_{}_agent_{}_V2I_{}_V2V_{}_Eve_{}.pt'.format(i_ep + 1, i, env.n_V2I, env.n_V2V, env.n_Eve))
            torch.save(current_agent.action_network.state_dict(), path)

    print('loss:',loss)
    print('reward:{}'.format(reward_ep))
    reward_record.append(reward_ep)
    
for i in range(env.n_V2V):
    current_agent = agent_list[i]
    #  + 'V2I_{}_V2V_{}_Eve_{}'.format(n_CUE, n_VUE, n_Eve)
    _, path = util.get_model_path('agent_{}_V2I_{}_V2V_{}_Eve_{}.pt'.format(i, env.n_V2I, env.n_V2V, env.n_Eve))
    torch.save(current_agent.action_network.state_dict(), path)
    print(path)
_, fig_path, reward_data_path = util.get_train_result_path('reward', 'reward_V2I_{}_V2V_{}_Eve_{}_discount{}'.format(env.n_V2I, env.n_V2V, env.n_Eve, discount))

plt.plot(reward_record, label='reward')
plt.legend()
plt.grid(True)
plt.savefig(fig_path)

plt.close()


np.save(reward_data_path, arr=np.asarray(reward_record))





