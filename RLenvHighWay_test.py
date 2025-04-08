import random

import numpy as np
from numpy.random import normal
from HighwayChannel import HwyChannelLargeScaleFadingGenerator, HwyChannelLargeScaleFadingGenerator2
import matplotlib.pyplot as plt
from munkres import Munkres
np.random.seed(1234)
random.seed(4321)
import math

class RLHighWayEnvironment():
    '''
    This class implement the simulator environment for reinforcement learning in the Highway scenario
    '''

    def __init__(self, config):
        '''
        Construction methods for class RLHighWayEnvironment
        :param config: dict containing key parameters for simulation
        '''
        self.p_mm = 23
        self.p_km = 23

        # transceiver configuration
        self.power_list_V2V_dB = config["powerV2VdB"]
        self.power_V2I_dB = config["powerV2I"]
        self.sig2_dB = config["backgroundNoisedB"]
        self.sig2 = 10 ** (self.sig2_dB / 10)
        # agent configuration
        self.n_V2I = config["n_V2I"]
        self.n_V2V = config["n_V2V"]
        self.n_Eve = config["n_Eve"]
        self.n_Channel = self.n_V2I
        # protocol configuration
        self.seed = config["seed"]
        self.time_fast_fading = 0.001
        self.time_slow_fading = 0.1
        self.bandwidth = 1e6
        self.demand_size = config["demand"]
        self.speed = config["speed"] / 3.6
        self.Eveknow = config["Eveknow"]
        self.n_power_level = len(self.power_list_V2V_dB)
        self.disBstoHwy = 35
        self.lambdda = 0.1
        self.speed_diff = config["speed_diff"] / 3.6
        self.POSSION = config["POSSION"]
        self.EveDist = config["EveDist"]
        # initialize vehicle and channel sampler
        self.lrg_generator = HwyChannelLargeScaleFadingGenerator()
        # internal simulation parameters
        # self.active_links = np.ones(self.numDUE, dtype="bool")
        # self.individual_time_limit = self.time_slow_fading * np.ones(self.numDUE)
        # self.demand = self.demand_size * np.ones(self.numDUE)
        self.d0 = np.sqrt(500 ** 2 - self.disBstoHwy ** 2)
        self.V2V_interference_dqn = np.zeros((self.n_V2V, self.n_V2I))
        self.V2E_interference_dqn = np.zeros((self.n_V2V, self.n_Eve, self.n_V2I))
        self.V2V_interference_ppo = np.zeros((self.n_V2V, self.n_V2I))
        self.V2E_interference_ppo = np.zeros((self.n_V2V, self.n_Eve, self.n_V2I))
        self.V2V_interference_ddpg = np.zeros((self.n_V2V, self.n_V2I))
        self.V2E_interference_ddpg = np.zeros((self.n_V2V, self.n_Eve, self.n_V2I))
        self.BS_position = [0, 0]
        self.numLane = 6
        self.laneWidth = 4
        self.time_block_limit = 10
        self.time_slow_fading = self.time_fast_fading * self.time_block_limit
        self.time_limit = self.time_block_limit / 1000
        self.munkres = Munkres()


    def init_simulation(self):
        '''
        Initialize Highway Environment simulator
        :return:
        '''
        self.generate_vehicles()
        # else:
        #     self.generate_vehicles_woPossion()
        self.update_V2VReceiver()
        # self.generate_vehicles()
        self.update_channels_slow()
        self.update_channels_fast()

        # ---------------ddpg-----------------
        self.active_links_ddpg = np.ones(self.n_V2V, dtype="bool")
        self.individual_time_limit_ddpg = self.time_limit * np.ones(self.n_V2V)
        self.demand_ddpg = self.demand_size * np.ones(self.n_V2V)

        # ---------------dqn-----------------
        self.active_links_dqn = np.ones(self.n_V2V, dtype="bool")
        self.individual_time_limit_dqn = self.time_limit * np.ones(self.n_V2V)
        self.demand_dqn = self.demand_size * np.ones(self.n_V2V)

        # ---------------ppo-----------------
        self.active_links_ppo = np.ones(self.n_V2V, dtype="bool")
        self.individual_time_limit_ppo = self.time_limit * np.ones(self.n_V2V)
        self.demand_ppo = self.demand_size * np.ones(self.n_V2V)

        # ---------------random-----------------
        self.active_links_rand = np.ones(self.n_V2V, dtype="bool")
        self.individual_time_limit_rand = self.time_limit * np.ones(self.n_V2V)
        self.demand_rand = self.demand_size * np.ones(self.n_V2V)

        # ---------------dpra-----------------
        self.active_links_dpra = np.ones(self.n_V2V, dtype="bool")
        self.individual_time_limit_dpra = self.time_limit * np.ones(self.n_V2V)
        self.demand_dpra = self.demand_size * np.ones(self.n_V2V)

        # self.vehSped = np.zeros(len(self.vehPos))
    def update_env_slow(self, POSITION=True):
        if POSITION == True:
            self.update_vehicle_position()
            self.update_V2VReceiver()
            self.update_channels_slow()
            self.update_channels_fast()
        # ---------------ddpg-----------------
        self.active_links_ddpg = np.ones(self.n_V2V, dtype="bool")
        self.individual_time_limit_ddpg = self.time_limit * np.ones(self.n_V2V)
        self.demand_ddpg = self.demand_size * np.ones(self.n_V2V)

        # ---------------dqn-----------------
        self.active_links_dqn = np.ones(self.n_V2V, dtype="bool")
        self.individual_time_limit_dqn = self.time_limit * np.ones(self.n_V2V)
        self.demand_dqn = self.demand_size * np.ones(self.n_V2V)

        # ---------------ppo-----------------
        self.active_links_ppo = np.ones(self.n_V2V, dtype="bool")
        self.individual_time_limit_ppo = self.time_limit * np.ones(self.n_V2V)
        self.demand_ppo = self.demand_size * np.ones(self.n_V2V)

        # ---------------random-----------------
        self.active_links_rand = np.ones(self.n_V2V, dtype="bool")
        self.individual_time_limit_rand = self.time_limit * np.ones(self.n_V2V)
        self.demand_rand = self.demand_size * np.ones(self.n_V2V)

        # ---------------dpra-----------------
        self.active_links_dpra = np.ones(self.n_V2V, dtype="bool")
        self.individual_time_limit_dpra = self.time_limit * np.ones(self.n_V2V)
        self.demand_dpra = self.demand_size * np.ones(self.n_V2V)

    def update_env_fast(self):
        self.update_channels_fast()

    
    def generate_vehicles(self):
        # 平均速度*2.5s 2.5*v
        d_avg = 2.5 * 100 / 3.6
        while 1:
            vehPos = []
            vehDir = []
            vehSep = []
            for ilane in range(self.numLane):
                nveh = int(2 * self.d0 / d_avg)
                posi_ilane = np.zeros((nveh, 2))
                posi_ilane[:, 0] = (2 * np.random.rand(nveh) - 1) * self.d0
                posi_ilane[:, 1] = self.disBstoHwy + (ilane * (self.laneWidth) + self.laneWidth / 2) * np.ones(nveh)
                vehPos.append(posi_ilane)
                if ilane < self.numLane // 2:
                    vehDir.append([0] * nveh)  # left -> right
                else:
                    vehDir.append([1] * nveh)  # right -> left
                vehSep.append([random.uniform(self.speed - self.speed_diff, self.speed)]*nveh)
            vehPos = np.concatenate(vehPos)
            vehDir = np.concatenate(vehDir)
            vehSep = np.concatenate(vehSep)
            numVeh = vehPos.shape[0]
            if numVeh > self.n_V2I + self.n_V2V * 2 + self.n_Eve:
                break
        # 从所有的汽车中随机选择一些车辆出来作为D2D通信的发起者和接收方，每个D2D通信的发起者的接收机都是距离它最近的那个
        indPerm = np.random.permutation(numVeh)
        indDUETransmitter = indPerm[:self.n_V2V]
        indDUEReceiver = -np.ones(self.n_V2V, dtype="int")
        for i in range(self.n_V2V):
            minDist = np.inf
            tmpInd = 0
            for j in range(numVeh):
                # if j in indDUETransmitter or j in indDUEReceiver:
                #     continue
                if j == indDUETransmitter[i]:
                    continue
                newDist = np.sqrt(np.sum((vehPos[indDUETransmitter[i]] - vehPos[j]) ** 2))
                if newDist < minDist:
                    tmpInd = j
                    minDist = newDist
            indDUEReceiver[i] = tmpInd
        # 从剩下的车里面随机选择一些作为CUE
        cntCUE = self.n_V2V + 1
        indCUE = []
        while abs(cntCUE) <= numVeh:
            if indPerm[cntCUE] not in indDUEReceiver:
                indCUE.append(indPerm[cntCUE])
            cntCUE += 1
            if len(indCUE) >= self.n_V2I:
                break

        evePos = []
        posEve = np.zeros((self.n_Eve, 2))
        loc_eve = [0, -25, 25]
        for i in range(self.n_Eve):
            posEve[i, 0] = loc_eve[i]
            posEve[i, 1] = self.disBstoHwy - self.EveDist
        evePos.append(posEve)
        evePos = np.concatenate(evePos)
        self.V2I = np.array(indCUE)
        self.V2VTransmitter = indDUETransmitter
        self.V2VReceiver = indDUEReceiver
        self.evePos = evePos
        self.vehPos = vehPos
        self.vehDir = vehDir
        self.vehSped = vehSep

    def update_vehicle_position(self):
        '''
        Update the position of each vehicle according to their current position, direction and speed
        :return:
        '''
        factor = 1

        for veh_idx in range(len(self.vehPos)):
            cur_posi = self.vehPos[veh_idx]
            speed = self.vehSped[veh_idx]
            if self.vehDir[veh_idx] == 0:  # left -> right
                tmp = cur_posi[0] + speed * factor * self.time_fast_fading * self.time_block_limit
                if tmp > self.d0:
                    self.vehPos[veh_idx][0] = tmp - 2 * self.d0
                else:
                    self.vehPos[veh_idx][0] = tmp
            else:
                tmp = cur_posi[0] - speed * factor * self.time_fast_fading * self.time_block_limit
                if tmp < -self.d0:
                    self.vehPos[veh_idx][0] = tmp + 2 * self.d0
                else:
                    self.vehPos[veh_idx][0] = tmp

    def update_V2VReceiver(self):
        '''
        Update the V2V receiver according the updated position of each vehicle
        :return:
        '''
        numVeh = len(self.vehPos)
        self.V2VReceiver = -np.ones(self.n_V2V, dtype="int")
        for i in range(self.n_V2V):
            minDist = np.inf
            tmpInd = 0
            for j in range(numVeh):
                # if j in self.V2VTransmitter or j in self.V2I:
                #     continue
                if j == self.V2VTransmitter[i]:
                    continue
                newDist = np.sqrt(np.sum((self.vehPos[self.V2VTransmitter[i]] - self.vehPos[j]) ** 2))
                if newDist < minDist:
                    tmpInd = j
                    minDist = newDist
            self.V2VReceiver[i] = tmpInd

    def update_channels_slow(self):

        self.V2I_channel_dB = np.zeros(self.n_V2I)
        self.V2V_channel_dB = np.zeros(self.n_V2V)
        self.V2V_V2I_interference_channel_dB = np.zeros(self.n_V2V)
        self.V2I_V2V_interference_channel_dB = np.zeros((self.n_V2I, self.n_V2V))
        self.V2V_V2V_channel_dB = np.zeros((self.n_V2V, self.n_V2V))
        self.Eve_channel_dB = np.zeros((self.n_V2V, self.n_Eve))
        self.V2I_Eve_interference_channel_dB = np.zeros((self.n_V2I, self.n_Eve))

        for m in range(self.n_V2I):
            # 计算第m个CUE对基站的距离和路损，假设基站的坐标是（0，0）
            dist_mB = np.sqrt(np.sum((self.vehPos[self.V2I[m]]) ** 2))  # m-th CUE到基站的距离
            self.V2I_channel_dB[m] = self.lrg_generator.generate_fading_V2I(dist_mB)
            # 计算第m个V2I和第k个V2V之间的距离、路损
            for k in range(self.n_V2V):
                pos1 = self.vehPos[self.V2I[m]]
                pos2 = self.vehPos[self.V2VReceiver[k]]
                dist_mk = np.sqrt(np.sum((pos1 - pos2) ** 2))
                self.V2I_V2V_interference_channel_dB[m, k] = self.lrg_generator.generate_fading_V2V(dist_mk)
            # 计算第m个V2I和第e个窃听者之间的距离、路损
            for e in range(self.n_Eve):
                pos1 = self.vehPos[self.V2I[m]]
                pos2 = self.evePos[e]
                dist_me = np.sqrt(np.sum((pos1 - pos2) ** 2))
                self.V2I_Eve_interference_channel_dB[m, e] = self.lrg_generator.generate_fading_VehicleEve(dist_me)

        for k in range(self.n_V2V):
            # 计算第K对DUE之间的距离和路损
            pos1 = self.vehPos[self.V2VTransmitter[k]]
            pos2 = self.vehPos[self.V2VReceiver[k]]
            # 计算第k对DUE的发射机对基站的干扰
            dist_kB = np.sqrt(np.sum((pos1) ** 2))  # k-th DUE发射机到基站的距离
            self.V2V_V2I_interference_channel_dB[k] = self.lrg_generator.generate_fading_V2I(dist_kB)
            for j in range(self.n_V2V):
                pos1 = self.vehPos[self.V2VTransmitter[k]]
                pos2 = self.vehPos[self.V2VReceiver[j]]
                dist_kj = np.sqrt(np.sum((pos1 - pos2) ** 2))
                self.V2V_V2V_channel_dB[k, j] = self.lrg_generator.generate_fading_V2V(dist_kj)

            for e in range(self.n_Eve):
                pos1 = self.vehPos[self.V2VTransmitter[k]]
                pos2 = self.evePos[e]
                dist_ke = np.sqrt(np.sum((pos1 - pos2) ** 2))
                self.Eve_channel_dB[k, e] = self.lrg_generator.generate_fading_VehicleEve(dist_ke)

    def update_channels_fast(self):
        '''
        Update fasting fading component for four kinds of channels
        :return:
        '''
        V2I_channel_fast_dB = np.repeat(self.V2I_channel_dB[:, np.newaxis], self.n_Channel, axis=1)
        fast_componet = np.abs(
            normal(0, 1, V2I_channel_fast_dB.shape) + 1j * normal(0, 1, V2I_channel_fast_dB.shape)) / np.sqrt(2)
        self.V2I_channel_with_fast_dB = V2I_channel_fast_dB + 20 * np.log10(fast_componet)


        V2I_V2V_interference_channel_with_fast_dB = np.repeat(self.V2I_V2V_interference_channel_dB[:, :, np.newaxis],
                                                              self.n_Channel, axis=2)
        fast_componet = np.abs(normal(0, 1, V2I_V2V_interference_channel_with_fast_dB.shape) + 1j * normal(0, 1,
                                                                                                           V2I_V2V_interference_channel_with_fast_dB.shape)) / np.sqrt(
            2)
        self.V2I_V2V_interference_channel_with_fast_dB = V2I_V2V_interference_channel_with_fast_dB + 20 * np.log10(
            fast_componet)

        V2V_V2V_interference_channel_with_fast_dB = np.repeat(self.V2V_V2V_channel_dB[:, :, np.newaxis],
                                                              self.n_Channel, axis=2)
        fast_componet = np.abs(normal(0, 1, V2V_V2V_interference_channel_with_fast_dB.shape) + 1j * normal(0, 1,
                                                                                                           V2V_V2V_interference_channel_with_fast_dB.shape)) / np.sqrt(
            2)
        self.V2V_V2V_channel_with_fast_dB = V2V_V2V_interference_channel_with_fast_dB + 20 * np.log10(
            fast_componet)

        V2V_V2I_interference_channel_with_fast_dB = np.repeat(self.V2V_V2I_interference_channel_dB[:, np.newaxis],
                                                              self.n_Channel, axis=1)
        fast_componet = np.abs(normal(0, 1, V2V_V2I_interference_channel_with_fast_dB.shape) + 1j * normal(0, 1,
                                                                                                           V2V_V2I_interference_channel_with_fast_dB.shape)) / np.sqrt(
            2)
        self.V2V_V2I_interference_channel_with_fast_dB = V2V_V2I_interference_channel_with_fast_dB + 20 * np.log10(
            fast_componet)

        Eve_channel_with_fast_dB = np.repeat(self.Eve_channel_dB[:, :, np.newaxis], self.n_Channel, axis=2)
        fast_componet = np.abs(
            normal(0, 1, Eve_channel_with_fast_dB.shape) + 1j * normal(0, 1, Eve_channel_with_fast_dB.shape)) / np.sqrt(
            2)
        self.Eve_channel_with_fast_dB = Eve_channel_with_fast_dB + 20 * np.log10(fast_componet)

        V2I_Eve_interference_channel_with_fast_dB = np.repeat(self.V2I_Eve_interference_channel_dB[:, :, np.newaxis],
                                                              self.n_Channel, axis=2)
        fast_componet = np.abs(normal(0, 1, V2I_Eve_interference_channel_with_fast_dB.shape) + 1j * normal(0, 1,
                                                                                                           V2I_Eve_interference_channel_with_fast_dB.shape)) / np.sqrt(
            2)
        self.V2I_Eve_interference_channel_with_fast_dB = V2I_Eve_interference_channel_with_fast_dB + 20 * np.log10(
            fast_componet)

    def get_state_dqn(self, idx=0, epsilon=0.02, ind_episode=1., mode='gain'):

        CSI_fast_state = []
        CSI_slow_state = []
        CSI_gain_state = []
        state = []
        # 收到的干扰
        V2V_interference = self.V2V_interference_dqn[idx, :]
        V2V_interference = (-V2V_interference - 60) / 60
        load_remaining = [self.demand_dqn[idx] / self.demand_size]
        time_remaining = [self.individual_time_limit_dqn[idx] / (self.time_block_limit / 1000)]

        payload_state = np.concatenate([
            np.array(load_remaining),  # 将 Python 列表转换为数组
            np.array(time_remaining),
            V2V_interference  # 将 Python 列表转换为数组
        ])

        if mode == 'gain':
            # 信道增益
            V2V_V2I_gain = self.V2V_V2I_interference_channel_with_fast_dB[idx, :]
            V2V_V2I_gain = (V2V_V2I_gain + 80) / 60

            V2I_V2V_gain = self.V2I_V2V_interference_channel_with_fast_dB[:, idx]
            V2I_V2V_gain = (V2I_V2V_gain + 80) / 60

            V2V_V2V_gain = self.V2V_V2V_channel_with_fast_dB[:, idx, :]
            V2V_V2V_gain = (V2V_V2V_gain + 80) / 60
            
            
            V2I_Eve_gain = np.zeros([self.n_V2I, self.n_V2I])
            for m in range(self.n_V2I):
                eveIdx = np.argmax(self.V2I_Eve_interference_channel_dB[m])
                V2I_Eve_gain[m] = self.V2I_Eve_interference_channel_with_fast_dB[m, eveIdx, :]
            V2I_Eve_gain = (V2I_Eve_gain + 80) / 60

            
            eveIdx = np.argmax(self.Eve_channel_dB[idx, :])
            Eve_gain = self.Eve_channel_with_fast_dB[idx, eveIdx]
            Eve_gain = (Eve_gain + 80) / 60


            CSI_gain_state = np.concatenate([
                V2V_V2I_gain,
                V2I_V2V_gain.reshape(-1),
                V2V_V2V_gain.reshape(-1),
                Eve_gain.reshape(-1),
                V2I_Eve_gain.reshape(-1)
            ])
            state = np.concatenate([CSI_gain_state, payload_state])
        else:
            # 快衰落
            V2V_V2I_fast = self.V2V_V2I_interference_channel_with_fast_dB[idx, :] - \
                           self.V2V_V2I_interference_channel_dB[
                               idx]  # h_kb
            V2I_V2V_fast = self.V2I_V2V_interference_channel_with_fast_dB[:,
                           idx] - self.V2I_V2V_interference_channel_dB[:,
                                  idx]  # h_mk
            V2V_V2V_fast = self.V2V_V2V_channel_with_fast_dB[:, idx, :] - self.V2V_V2V_channel_dB[:, idx].reshape(
                (self.n_V2V, 1))  # h_k, h_k'k
            # 标准化
            V2V_V2V_fast = (V2V_V2V_fast + 10) / 35  # normalize the fast fading component of V2V links
            V2V_V2I_fast = (V2V_V2I_fast + 10) / 35
            V2I_V2V_fast = (V2I_V2V_fast + 10) / 35

            # 慢衰落
            V2V_V2I_slow = self.V2V_V2I_interference_channel_dB[idx]  # h_kb
            V2I_V2V_slow = self.V2I_V2V_interference_channel_dB[:, idx]  # h_mk
            V2V_V2V_slow = self.V2V_V2V_channel_dB[:, idx]  # h_k, h_k'k
            # 标准化
            V2V_V2V_slow = (V2V_V2V_slow + 80) / 60
            V2V_V2I_slow = (V2V_V2I_slow + 80) / 60
            V2I_V2V_slow = (V2I_V2V_slow + 80) / 60

            # V2E_interference = self.V2E_interference[idx, :, :]
            # V2E_interference = (-V2E_interference - 60) / 60

            # 窃听链路
            V2I_Eve_fast = self.V2I_Eve_interference_channel_with_fast_dB - self.V2I_Eve_interference_channel_dB.reshape(
                self.n_V2I, self.n_Eve, 1)  # h_me
            Eve_fast = self.Eve_channel_with_fast_dB[idx, :, :] - self.Eve_channel_dB[idx, :].reshape(
                (self.n_Eve, 1))  # h_ke
            V2I_Eve_fast = (V2I_Eve_fast + 10) / 35  # normalize the fast fading component of V2I links
            Eve_fast = (Eve_fast + 10) / 35  # normalize the fast fading component of V2V links

            V2I_Eve_slow = self.V2I_Eve_interference_channel_dB  # h_me
            Eve_slow = self.Eve_channel_dB[idx, :]  # h_ke
            V2I_Eve_slow = (V2I_Eve_slow + 80) / 60
            Eve_slow = (Eve_slow + 80) / 60

            CSI_fast_state = np.concatenate([
                V2V_V2I_fast,
                V2I_V2V_fast.reshape(-1),
                V2V_V2V_fast.reshape(-1),
                Eve_fast.reshape(-1),
                V2I_Eve_fast.reshape(-1)
            ])

            CSI_slow_state = np.concatenate([
                np.array([V2V_V2I_slow]),
                V2I_V2V_slow,
                V2V_V2V_slow,
                V2I_Eve_slow.reshape(-1),
                Eve_slow,
            ])
            state = np.concatenate([CSI_fast_state, CSI_slow_state, payload_state])

        return state




    def get_state_ddpg(self, idx=0, epsilon=0.02, ind_episode=1., mode='gain'):

        CSI_fast_state = []
        CSI_slow_state = []
        CSI_gain_state = []
        state = []
        # 收到的干扰
        V2V_interference = self.V2V_interference_ddpg[idx, :]
        V2V_interference = (-V2V_interference - 60) / 60
        load_remaining = [self.demand_ddpg[idx] / self.demand_size]
        time_remaining = [self.individual_time_limit_ddpg[idx] / (self.time_block_limit / 1000)]

        payload_state = np.concatenate([
            np.array(load_remaining),  # 将 Python 列表转换为数组
            np.array(time_remaining),
            V2V_interference  # 将 Python 列表转换为数组
        ])

        if mode == 'gain':
            # 信道增益
            V2V_V2I_gain = self.V2V_V2I_interference_channel_with_fast_dB[idx, :]
            V2V_V2I_gain = (V2V_V2I_gain + 80) / 60

            V2I_V2V_gain = self.V2I_V2V_interference_channel_with_fast_dB[:, idx, :]
            V2I_V2V_gain = (V2I_V2V_gain + 80) / 60

            V2V_V2V_gain = self.V2V_V2V_channel_with_fast_dB[:, idx, :]
            V2V_V2V_gain = (V2V_V2V_gain + 80) / 60

            V2I_Eve_gain = np.zeros([self.n_V2I, self.n_V2I])
            for m in range(self.n_V2I):
                eveIdx = np.argmax(self.V2I_Eve_interference_channel_dB[m])
                V2I_Eve_gain[m] = self.V2I_Eve_interference_channel_with_fast_dB[m, eveIdx, :]
            V2I_Eve_gain = (V2I_Eve_gain + 80) / 60

            
            eveIdx = np.argmax(self.Eve_channel_dB[idx, :])
            Eve_gain = self.Eve_channel_with_fast_dB[idx, eveIdx]
            Eve_gain = (Eve_gain + 80) / 60

            CSI_gain_state = np.concatenate([
                V2V_V2I_gain,
                V2I_V2V_gain.reshape(-1),
                V2V_V2V_gain.reshape(-1),
                Eve_gain.reshape(-1),
                V2I_Eve_gain.reshape(-1)
            ])
            state = np.concatenate([CSI_gain_state, payload_state])
        else:
            # 快衰落
            V2V_V2I_fast = self.V2V_V2I_interference_channel_with_fast_dB[idx, :] - \
                           self.V2V_V2I_interference_channel_dB[
                               idx]  # h_kb
            V2I_V2V_fast = self.V2I_V2V_interference_channel_with_fast_dB[:,
                           idx] - self.V2I_V2V_interference_channel_dB[:,
                                  idx]  # h_mk
            V2V_V2V_fast = self.V2V_V2V_channel_with_fast_dB[:, idx, :] - self.V2V_V2V_channel_dB[:, idx].reshape(
                (self.n_V2V, 1))  # h_k, h_k'k
            # 标准化
            V2V_V2V_fast = (V2V_V2V_fast + 10) / 35  # normalize the fast fading component of V2V links
            V2V_V2I_fast = (V2V_V2I_fast + 10) / 35
            V2I_V2V_fast = (V2I_V2V_fast + 10) / 35

            # 慢衰落
            V2V_V2I_slow = self.V2V_V2I_interference_channel_dB[idx]  # h_kb
            V2I_V2V_slow = self.V2I_V2V_interference_channel_dB[:, idx]  # h_mk
            V2V_V2V_slow = self.V2V_V2V_channel_dB[:, idx]  # h_k, h_k'k
            # 标准化
            V2V_V2V_slow = (V2V_V2V_slow + 80) / 60
            V2V_V2I_slow = (V2V_V2I_slow + 80) / 60
            V2I_V2V_slow = (V2I_V2V_slow + 80) / 60

            # V2E_interference = self.V2E_interference[idx, :, :]
            # V2E_interference = (-V2E_interference - 60) / 60

            # 窃听链路
            V2I_Eve_fast = self.V2I_Eve_interference_channel_with_fast_dB - self.V2I_Eve_interference_channel_dB.reshape(
                self.n_V2I, self.n_Eve, 1)  # h_me
            Eve_fast = self.Eve_channel_with_fast_dB[idx, :, :] - self.Eve_channel_dB[idx, :].reshape(
                (self.n_Eve, 1))  # h_ke
            V2I_Eve_fast = (V2I_Eve_fast + 10) / 35  # normalize the fast fading component of V2I links
            Eve_fast = (Eve_fast + 10) / 35  # normalize the fast fading component of V2V links

            V2I_Eve_slow = self.V2I_Eve_interference_channel_dB  # h_me
            Eve_slow = self.Eve_channel_dB[idx, :]  # h_ke
            V2I_Eve_slow = (V2I_Eve_slow + 80) / 60
            Eve_slow = (Eve_slow + 80) / 60

            CSI_fast_state = np.concatenate([
                V2V_V2I_fast,
                V2I_V2V_fast.reshape(-1),
                V2V_V2V_fast.reshape(-1),
                Eve_fast.reshape(-1),
                V2I_Eve_fast.reshape(-1)
            ])

            CSI_slow_state = np.concatenate([
                np.array([V2V_V2I_slow]),
                V2I_V2V_slow,
                V2V_V2V_slow,
                V2I_Eve_slow.reshape(-1),
                Eve_slow,
            ])
            state = np.concatenate([CSI_fast_state, CSI_slow_state, payload_state])

        return state

    def get_gstate(self):
        state = []
        for i in range(self.n_V2V):
            state.append(self.get_state_dqn(i))
        state = np.concatenate(state)
        return state


    def compute_V2V_interference(self, action, mode='dqn'):
        active_links = np.zeros(self.n_V2V, dtype='bool')
        if mode == 'rand':
            active_links = self.active_links_rand.copy()
        elif mode == 'dqn':
            active_links = self.active_links_dqn.copy()
        elif mode == 'ppo':
            active_links = self.active_links_ppo.copy()
        RB_selection = action[:, 0]
        power_selection = action[:, 1]
        # interference from V2I link
        V2V_interference = np.zeros((self.n_V2V, self.n_V2I)) + self.sig2
        for i in range(self.n_V2V):
            # if current V2V link has finished its transmission, it then turns to silent mode
            if not active_links[i]:
                continue
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            V2I_power_dB = self.power_V2I_dB + self.V2I_V2V_interference_channel_with_fast_dB[RB_i, i, RB_i]
            V2V_interference[i, RB_i] += 10 ** (V2I_power_dB / 10)
            # interference from other V2V link with the same RB
            for j in range(self.n_V2V):
                RB_j = RB_selection[j]
                power_dB_j = power_selection[j]
                if not active_links[j] or j == i:
                    continue
                if RB_j == RB_i:
                    power_j2i = self.power_list_V2V_dB[power_dB_j] + self.V2V_V2V_channel_with_fast_dB[
                        j, i, RB_j]
                    V2V_interference[i, RB_j] += 10 ** (power_j2i / 10)
        # transform inference from linear scale to dB

        # if mode == 'rand':
        #     active_links = self.active_links_rand.copy()
        if mode == 'dqn':
            self.V2V_interference_dqn = 10 * np.log10(V2V_interference)
        elif mode == 'ppo':
            self.V2V_interference_ppo = 10 * np.log10(V2V_interference)

        V2E_interference = np.zeros((self.n_V2V, self.n_Eve, self.n_V2I)) + self.sig2
        for i in range(self.n_V2V):
            # if current V2V link has finished its transmission, it then turns to silent mode
            if not active_links[i]:
                continue
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            V2I_power_dB = self.power_V2I_dB + self.V2I_Eve_interference_channel_with_fast_dB[RB_i, :, RB_i]
            V2E_interference[i, :, RB_i] += 10 ** (V2I_power_dB / 10)
            # interference from other V2V link with the same RB
            for j in range(self.n_V2V):
                RB_j = RB_selection[j]
                power_dB_j = power_selection[j]
                if not active_links[j] or j == i:
                    continue
                if RB_j == RB_i:
                    power_j2i = self.power_list_V2V_dB[power_dB_j] + self.Eve_channel_with_fast_dB[j, :, RB_j]
                    V2E_interference[i, :, RB_j] += 10 ** (power_j2i / 10)
        # transform inference from linear scale to dB
        self.V2E_interference_dqn = 10 * np.log10(V2E_interference)


    def compute_V2V_interference_dqn(self, action):
        RB_selection = action[:, 0]
        power_selection = action[:, 1]
        # interference from V2I link
        V2V_interference = np.zeros((self.n_V2V, self.n_V2I)) + self.sig2
        for i in range(self.n_V2V):
            # if current V2V link has finished its transmission, it then turns to silent mode
            if not self.active_links_dqn[i]:
                continue
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            V2I_power_dB = self.power_V2I_dB + self.V2I_V2V_interference_channel_with_fast_dB[RB_i, i, RB_i]
            V2V_interference[i, RB_i] += 10 ** (V2I_power_dB / 10)
            # interference from other V2V link with the same RB
            for j in range(self.n_V2V):
                RB_j = RB_selection[j]
                power_dB_j = power_selection[j]
                if not self.active_links_dqn[j] or j == i:
                    continue
                if RB_j == RB_i:
                    power_j2i = self.power_list_V2V_dB[power_dB_j] + self.V2V_V2V_channel_with_fast_dB[
                        j, i, RB_j]
                    V2V_interference[i, RB_j] += 10 ** (power_j2i / 10)
        # transform inference from linear scale to dB
        self.V2V_interference_dqn = 10 * np.log10(V2V_interference)

        V2E_interference = np.zeros((self.n_V2V, self.n_Eve, self.n_V2I)) + self.sig2
        for i in range(self.n_V2V):
            # if current V2V link has finished its transmission, it then turns to silent mode
            if not self.active_links_dqn[i]:
                continue
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            V2I_power_dB = self.power_V2I_dB + self.V2I_Eve_interference_channel_with_fast_dB[RB_i, :, RB_i]
            V2E_interference[i, :, RB_i] += 10 ** (V2I_power_dB / 10)
            # interference from other V2V link with the same RB
            for j in range(self.n_V2V):
                RB_j = RB_selection[j]
                power_dB_j = power_selection[j]
                if not self.active_links_dqn[j] or j == i:
                    continue
                if RB_j == RB_i:
                    power_j2i = self.power_list_V2V_dB[power_dB_j] + self.Eve_channel_with_fast_dB[j, :, RB_j]
                    V2E_interference[i, :, RB_j] += 10 ** (power_j2i / 10)
        # transform inference from linear scale to dB
        self.V2E_interference_dqn = 10 * np.log10(V2E_interference)

    def compute_V2V_interference_ppo(self, action):
        RB_selection = action[:, 0]
        power_selection = action[:, 1]
        # interference from V2I link
        V2V_interference = np.zeros((self.n_V2V, self.n_V2I)) + self.sig2
        for i in range(self.n_V2V):
            # if current V2V link has finished its transmission, it then turns to silent mode
            if not self.active_links_ppo[i]:
                continue
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            V2I_power_dB = self.power_V2I_dB + self.V2I_V2V_interference_channel_with_fast_dB[RB_i, i, RB_i]
            V2V_interference[i, RB_i] += 10 ** (V2I_power_dB / 10)
            # interference from other V2V link with the same RB
            for j in range(self.n_V2V):
                RB_j = RB_selection[j]
                power_dB_j = power_selection[j]
                if not self.active_links_ppo[j] or j == i:
                    continue
                if RB_j == RB_i:
                    power_j2i = self.power_list_V2V_dB[power_dB_j] + self.V2V_V2V_channel_with_fast_dB[
                        j, i, RB_j]
                    V2V_interference[i, RB_j] += 10 ** (power_j2i / 10)
        # transform inference from linear scale to dB
        self.V2V_interference_ppo = 10 * np.log10(V2V_interference)

        V2E_interference = np.zeros((self.n_V2V, self.n_Eve, self.n_V2I)) + self.sig2
        for i in range(self.n_V2V):
            # if current V2V link has finished its transmission, it then turns to silent mode
            if not self.active_links_ppo[i]:
                continue
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            V2I_power_dB = self.power_V2I_dB + self.V2I_Eve_interference_channel_with_fast_dB[RB_i, :, RB_i]
            V2E_interference[i, :, RB_i] += 10 ** (V2I_power_dB / 10)
            # interference from other V2V link with the same RB
            for j in range(self.n_V2V):
                RB_j = RB_selection[j]
                power_dB_j = power_selection[j]
                if not self.active_links_ppo[j] or j == i:
                    continue
                if RB_j == RB_i:
                    power_j2i = self.power_list_V2V_dB[power_dB_j] + self.Eve_channel_with_fast_dB[j, :, RB_j]
                    V2E_interference[i, :, RB_j] += 10 ** (power_j2i / 10)
        # transform inference from linear scale to dB
        self.V2E_interference_ppo = 10 * np.log10(V2E_interference)

    def compute_V2V_interference_ddpg(self, RB, PW):

        RB_selection = RB
        power_selection = PW
        # interference from V2I link
        V2V_interference = np.zeros((self.n_V2V, self.n_V2I)) + self.sig2
        for i in range(self.n_V2V):
            # if current V2V link has finished its transmission, it then turns to silent mode
            if not self.active_links_ddpg[i]:
                continue
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            V2I_power_dB = self.power_V2I_dB + self.V2I_V2V_interference_channel_with_fast_dB[RB_i, i, RB_i]
            V2V_interference[i, RB_i] += 10 ** (V2I_power_dB / 10)
            # interference from other V2V link with the same RB
            for j in range(self.n_V2V):
                RB_j = RB_selection[j]
                power_dB_j = power_selection[j]
                if not self.active_links_ddpg[j] or j == i:
                    continue
                if RB_j == RB_i:
                    power_j2i = power_dB_j + self.V2V_V2V_channel_with_fast_dB[j, i, RB_j]
                    V2V_interference[i, RB_j] += 10 ** (power_j2i / 10)
        # transform inference from linear scale to dB
        self.V2V_interference_ddpg = 10 * np.log10(V2V_interference)

        V2E_interference = np.zeros((self.n_V2V, self.n_Eve, self.n_V2I)) + self.sig2
        for i in range(self.n_V2V):
            # if current V2V link has finished its transmission, it then turns to silent mode
            if not self.active_links_ddpg[i]:
                continue
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            V2I_power_dB = self.power_V2I_dB + self.V2I_Eve_interference_channel_with_fast_dB[RB_i, :, RB_i]
            V2E_interference[i, :, RB_i] += 10 ** (V2I_power_dB / 10)
            # interference from other V2V link with the same RB
            for j in range(self.n_V2V):
                RB_j = RB_selection[j]
                power_dB_j = power_selection[j]
                if not self.active_links_ddpg[j] or j == i:
                    continue
                if RB_j == RB_i:
                    power_j2i = power_dB_j + self.Eve_channel_with_fast_dB[j, :, RB_j]
                    V2E_interference[i, :, RB_j] += 10 ** (power_j2i / 10)
        # transform inference from linear scale to dB
        self.V2E_interference_ddpg = 10 * np.log10(V2E_interference)


    def compute_rate_dqn(self, action):

        RB_selection = action[:, 0]
        power_selection = action[:, 1]
        # compute V2I rate
        V2I_interference = np.zeros(self.n_V2I)
        for i in range(self.n_V2V):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            if not self.active_links_dqn[i]:
                continue
            interference = self.power_list_V2V_dB[power_dB_i] + self.V2V_V2I_interference_channel_with_fast_dB[i, RB_i]
            V2I_interference[RB_i] += 10 ** (interference / 10)
        V2I_interference += self.sig2
        V2I_power_dB = self.power_V2I_dB + self.V2I_channel_with_fast_dB.diagonal()
        V2I_power = 10 ** (V2I_power_dB / 10)
        V2I_rate = np.log2(1 + np.divide(V2I_power, V2I_interference))

        # compute V2V rate
        V2V_interference = np.zeros(self.n_V2V)
        V2V_signal = np.zeros(self.n_V2V)
        for i in range(self.n_V2V):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            if not self.active_links_dqn[i]:
                continue
            # compute receiver signal strength for current V2V link
            receiver_power_dB = self.power_list_V2V_dB[power_dB_i] + self.V2V_V2V_channel_with_fast_dB[i, i, RB_i]
            V2V_signal[i] = 10 ** (receiver_power_dB / 10)
            # compute V2I link interference to current V2V link
            V2I_interference_power_dB = self.power_V2I_dB + self.V2I_V2V_interference_channel_with_fast_dB[
                RB_i, i, RB_i]
            V2V_interference[i] += 10 ** (V2I_interference_power_dB / 10)
            for k in range(self.n_V2V):
                if not self.active_links_dqn[k] or i == k:
                    continue
                RB_k = RB_selection[k]
                power_dB_k = power_selection[k]
                if RB_k == RB_i:
                    power_k2i = self.power_list_V2V_dB[power_dB_k] + self.V2V_V2V_channel_with_fast_dB[k, i, RB_k]
                    V2V_interference[i] += 10 ** (power_k2i / 10)
        V2V_interference += self.sig2
        V2V_rate = np.log2(1 + np.divide(V2V_signal, V2V_interference))

        Eve_interference = np.zeros((self.n_V2V, self.n_Eve))
        Eve_signal = np.zeros((self.n_V2V, self.n_Eve))
        for i in range(self.n_V2V):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            if not self.active_links_dqn[i]:
                continue
            # compute receiver signal strength for current V2V link
            receiver_power_dB = self.power_list_V2V_dB[power_dB_i] + self.Eve_channel_with_fast_dB[i, :, RB_i]
            Eve_signal[i] = 10 ** (receiver_power_dB / 10)

            V2I_Eve_interference_power_dB = self.power_V2I_dB + self.V2I_Eve_interference_channel_with_fast_dB[RB_i, :,
                                                                RB_i]
            Eve_interference[i] += 10 ** (V2I_Eve_interference_power_dB / 10)
            for k in range(self.n_V2V):
                if not self.active_links_dqn[k] or i == k:
                    continue
                RB_k = RB_selection[k]
                power_dB_k = power_selection[k]
                if RB_k == RB_i:
                    power_k2i = self.power_list_V2V_dB[power_dB_k] + self.Eve_channel_with_fast_dB[k, :, RB_k]
                    Eve_interference[i] += 10 ** (power_k2i / 10)
        Eve_interference += self.sig2
        Eve_rate = np.log2(1 + np.divide(Eve_signal, Eve_interference))
        Secrecy_rate = V2V_rate - np.max(Eve_rate, axis=-1)
        Secrecy_rate[Secrecy_rate <= 0] = 0

        return V2I_rate, V2V_rate, Eve_rate, Secrecy_rate

    def compute_rate_ppo(self, action):

        RB_selection = action[:, 0]
        power_selection = action[:, 1]
        # compute V2I rate
        V2I_interference = np.zeros(self.n_V2I)
        for i in range(self.n_V2V):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            if not self.active_links_ppo[i]:
                continue
            interference = self.power_list_V2V_dB[power_dB_i] + self.V2V_V2I_interference_channel_with_fast_dB[i, RB_i]
            V2I_interference[RB_i] += 10 ** (interference / 10)
        V2I_interference += self.sig2
        V2I_power_dB = self.power_V2I_dB + self.V2I_channel_with_fast_dB.diagonal()
        V2I_power = 10 ** (V2I_power_dB / 10)
        V2I_rate = np.log2(1 + np.divide(V2I_power, V2I_interference))

        # compute V2V rate
        V2V_interference = np.zeros(self.n_V2V)
        V2V_signal = np.zeros(self.n_V2V)
        for i in range(self.n_V2V):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            if not self.active_links_ppo[i]:
                continue
            # compute receiver signal strength for current V2V link
            receiver_power_dB = self.power_list_V2V_dB[power_dB_i] + self.V2V_V2V_channel_with_fast_dB[i, i, RB_i]
            V2V_signal[i] = 10 ** (receiver_power_dB / 10)
            # compute V2I link interference to current V2V link
            V2I_interference_power_dB = self.power_V2I_dB + self.V2I_V2V_interference_channel_with_fast_dB[
                RB_i, i, RB_i]
            V2V_interference[i] += 10 ** (V2I_interference_power_dB / 10)
            for k in range(self.n_V2V):
                if not self.active_links_ppo[k] or i == k:
                    continue
                RB_k = RB_selection[k]
                power_dB_k = power_selection[k]
                if RB_k == RB_i:
                    power_k2i = self.power_list_V2V_dB[power_dB_k] + self.V2V_V2V_channel_with_fast_dB[k, i, RB_k]
                    V2V_interference[i] += 10 ** (power_k2i / 10)
        V2V_interference += self.sig2
        V2V_rate = np.log2(1 + np.divide(V2V_signal, V2V_interference))

        Eve_interference = np.zeros((self.n_V2V, self.n_Eve))
        Eve_signal = np.zeros((self.n_V2V, self.n_Eve))
        for i in range(self.n_V2V):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            if not self.active_links_ppo[i]:
                continue
            # compute receiver signal strength for current V2V link
            receiver_power_dB = self.power_list_V2V_dB[power_dB_i] + self.Eve_channel_with_fast_dB[i, :, RB_i]
            Eve_signal[i] = 10 ** (receiver_power_dB / 10)

            V2I_Eve_interference_power_dB = self.power_V2I_dB + self.V2I_Eve_interference_channel_with_fast_dB[RB_i, :,
                                                                RB_i]
            Eve_interference[i] += 10 ** (V2I_Eve_interference_power_dB / 10)
            for k in range(self.n_V2V):
                if not self.active_links_ppo[k] or i == k:
                    continue
                RB_k = RB_selection[k]
                power_dB_k = power_selection[k]
                if RB_k == RB_i:
                    power_k2i = self.power_list_V2V_dB[power_dB_k] + self.Eve_channel_with_fast_dB[k, :, RB_k]
                    Eve_interference[i] += 10 ** (power_k2i / 10)
        Eve_interference += self.sig2
        Eve_rate = np.log2(1 + np.divide(Eve_signal, Eve_interference))
        Secrecy_rate = V2V_rate - np.max(Eve_rate, axis=-1)
        Secrecy_rate[Secrecy_rate <= 0] = 0

        return V2I_rate, V2V_rate, Eve_rate, Secrecy_rate


    def compute_rate_rand(self, action):
        RB_selection = action[:, 0]
        power_selection = action[:, 1]
        # compute V2I rate
        V2I_interference = np.zeros(self.n_V2I)
        for i in range(self.n_V2V):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            if not self.active_links_rand[i]:
                continue
            interference = self.power_list_V2V_dB[power_dB_i] + self.V2V_V2I_interference_channel_with_fast_dB[i, RB_i]
            V2I_interference[RB_i] += 10 ** (interference / 10)
        V2I_interference += self.sig2
        V2I_power_dB = self.power_V2I_dB + self.V2I_channel_with_fast_dB.diagonal()
        V2I_power = 10 ** (V2I_power_dB / 10)
        V2I_rate = np.log2(1 + np.divide(V2I_power, V2I_interference))

        # compute V2V rate
        V2V_interference = np.zeros(self.n_V2V)
        V2V_signal = np.zeros(self.n_V2V)
        for i in range(self.n_V2V):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            if not self.active_links_rand[i]:
                continue
            # compute receiver signal strength for current V2V link
            receiver_power_dB = self.power_list_V2V_dB[power_dB_i] + self.V2V_V2V_channel_with_fast_dB[i, i, RB_i]
            V2V_signal[i] = 10 ** (receiver_power_dB / 10)
            # compute V2I link interference to current V2V link
            V2I_interference_power_dB = self.power_V2I_dB + self.V2I_V2V_interference_channel_with_fast_dB[
                RB_i, i, RB_i]
            V2V_interference[i] += 10 ** (V2I_interference_power_dB / 10)
            for k in range(self.n_V2V):
                if not self.active_links_rand[k] or i == k:
                    continue
                RB_k = RB_selection[k]
                power_dB_k = power_selection[k]
                if RB_k == RB_i:
                    power_k2i = self.power_list_V2V_dB[power_dB_k] + self.V2V_V2V_channel_with_fast_dB[k, i, RB_k]
                    V2V_interference[i] += 10 ** (power_k2i / 10)
        V2V_interference += self.sig2
        V2V_rate = np.log2(1 + np.divide(V2V_signal, V2V_interference))

        Eve_interference = np.zeros((self.n_V2V, self.n_Eve))
        Eve_signal = np.zeros((self.n_V2V, self.n_Eve))
        for i in range(self.n_V2V):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            if not self.active_links_rand[i]:
                continue
            # compute receiver signal strength for current V2V link
            receiver_power_dB = self.power_list_V2V_dB[power_dB_i] + self.Eve_channel_with_fast_dB[i, :, RB_i]
            Eve_signal[i] = 10 ** (receiver_power_dB / 10)

            V2I_Eve_interference_power_dB = self.power_V2I_dB + self.V2I_Eve_interference_channel_with_fast_dB[RB_i, :,
                                                                RB_i]
            Eve_interference[i] += 10 ** (V2I_Eve_interference_power_dB / 10)
            for k in range(self.n_V2V):
                if not self.active_links_rand[k] or i == k:
                    continue
                RB_k = RB_selection[k]
                power_dB_k = power_selection[k]
                if RB_k == RB_i:
                    power_k2i = self.power_list_V2V_dB[power_dB_k] + self.Eve_channel_with_fast_dB[k, :, RB_k]
                    Eve_interference[i] += 10 ** (power_k2i / 10)
        Eve_interference += self.sig2
        Eve_rate = np.log2(1 + np.divide(Eve_signal, Eve_interference))
        Secrecy_rate = V2V_rate - np.max(Eve_rate, axis=-1)
        Secrecy_rate[Secrecy_rate <= 0] = 0

        return V2I_rate, V2V_rate, Eve_rate, Secrecy_rate


    def compute_rate_ddpg(self, RB, PW):

        RB_selection = RB
        power_selection = PW
        # compute V2I rate
        V2I_interference = np.zeros(self.n_V2I)
        for i in range(self.n_V2V):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            if not self.active_links_ddpg[i]:
                continue
            interference = power_dB_i + self.V2V_V2I_interference_channel_with_fast_dB[i, RB_i]
            V2I_interference[RB_i] += 10 ** (interference / 10)
        V2I_interference += self.sig2
        V2I_power_dB = self.power_V2I_dB + self.V2I_channel_with_fast_dB.diagonal()
        V2I_power = 10 ** (V2I_power_dB / 10)
        V2I_rate = np.log2(1 + np.divide(V2I_power, V2I_interference))

        # compute V2V rate
        V2V_interference = np.zeros(self.n_V2V)
        V2V_signal = np.zeros(self.n_V2V)
        for i in range(self.n_V2V):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            if not self.active_links_ddpg[i]:
                continue
            # compute receiver signal strength for current V2V link
            receiver_power_dB = power_dB_i + self.V2V_V2V_channel_with_fast_dB[i, i, RB_i]
            V2V_signal[i] = 10 ** (receiver_power_dB / 10)
            # compute V2I link interference to current V2V link
            V2I_interference_power_dB = self.power_V2I_dB + self.V2I_V2V_interference_channel_with_fast_dB[
                RB_i, i, RB_i]
            V2V_interference[i] += 10 ** (V2I_interference_power_dB / 10)
            for k in range(self.n_V2V):
                if not self.active_links_ddpg[k] or i == k:
                    continue
                RB_k = RB_selection[k]
                power_dB_k = power_selection[k]
                if RB_k == RB_i:
                    power_k2i = power_dB_k + self.V2V_V2V_channel_with_fast_dB[k, i, RB_k]
                    V2V_interference[i] += 10 ** (power_k2i / 10)
        V2V_interference += self.sig2
        V2V_rate = np.log2(1 + np.divide(V2V_signal, V2V_interference))

        Eve_interference = np.zeros((self.n_V2V, self.n_Eve))
        Eve_signal = np.zeros((self.n_V2V, self.n_Eve))
        for i in range(self.n_V2V):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            if not self.active_links_ddpg[i]:
                continue
            # compute receiver signal strength for current V2V link
            receiver_power_dB = power_dB_i + self.Eve_channel_with_fast_dB[i, :, RB_i]
            Eve_signal[i] = 10 ** (receiver_power_dB / 10)

            V2I_Eve_interference_power_dB = self.power_V2I_dB + self.V2I_Eve_interference_channel_with_fast_dB[RB_i, :,
                                                                RB_i]
            Eve_interference[i] += 10 ** (V2I_Eve_interference_power_dB / 10)
            for k in range(self.n_V2V):
                if not self.active_links_ddpg[k] or i == k:
                    continue
                RB_k = RB_selection[k]
                power_dB_k = power_selection[k]
                if RB_k == RB_i:
                    power_k2i = power_dB_k + self.Eve_channel_with_fast_dB[k, :, RB_k]
                    Eve_interference[i] += 10 ** (power_k2i / 10)
        Eve_interference += self.sig2
        Eve_rate = np.log2(1 + np.divide(Eve_signal, Eve_interference))
        Secrecy_rate = V2V_rate - np.max(Eve_rate, axis=-1)
        Secrecy_rate[Secrecy_rate <= 0] = 0

        return V2I_rate, V2V_rate, Eve_rate, Secrecy_rate


    def compute_rate(self, action, mode='rand'):
        active_links = np.zeros(self.n_V2V, dtype='bool')
        if mode == 'rand':
            active_links = self.active_links_rand.copy()
        elif mode == 'dqn':
            active_links = self.active_links_dqn.copy()
        elif mode == 'ppo':
            active_links = self.active_links_ppo.copy()
        elif mode == 'dpra':
            active_links = self.active_links_dpra.copy()

        RB_selection = action[:, 0]
        power_selection = action[:, 1]
        # ----------------compute V2I rate-----------------------
        V2I_interference = np.zeros(self.n_V2I)
        for i in range(self.n_V2V):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            if not active_links[i]:
                continue
            # 来自V2V的干扰_OK, V2V功率到基站之间的衰落 self.V2V_V2I_interference_channel_dB[k] = self.lrg_generator.generate_fading_V2I(dist_kB)
            interference = self.power_list_V2V_dB[power_dB_i] + self.V2V_V2I_interference_channel_with_fast_dB[i, RB_i]
            # 加干扰_OK
            V2I_interference[RB_i] += 10 ** (interference / 10)
        V2I_interference += self.sig2
        # V2I的信号强度
        # dist_mB = np.sqrt(np.sum((self.vehPos[self.V2I[m]]) ** 2))  # m-th CUE到基站的距离
        #             self.V2I_channel_dB[m] = self.lrg_generator.generate_fading_V2I(dist_mB)
        V2I_power_dB = self.power_V2I_dB + self.V2I_channel_with_fast_dB.diagonal()
        V2I_power = 10 ** (V2I_power_dB / 10)
        V2I_rate = np.log2(1 + np.divide(V2I_power, V2I_interference))

        # ----------------compute V2V rate-----------------------
        V2V_interference = np.zeros(self.n_V2V)
        V2V_signal = np.zeros(self.n_V2V)
        for i in range(self.n_V2V):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            if not active_links[i]:
                continue
            # compute receiver signal strength for current V2V link
            #             for j in range(self.n_V2V):
            #                 pos1 = self.vehPos[self.V2VTransmitter[k]]
            #                 pos2 = self.vehPos[self.V2VReceiver[j]]
            #                 dist_kj = np.sqrt(np.sum((pos1 - pos2) ** 2))
            #                 self.V2V_V2V_channel_dB[k, j] = self.lrg_generator.generate_fading_V2V(dist_kj)
            receiver_power_dB = self.power_list_V2V_dB[power_dB_i] + self.V2V_V2V_channel_with_fast_dB[i, i, RB_i]
            V2V_signal[i] = 10 ** (receiver_power_dB / 10)
            # compute V2I link interference to current V2V link
            #             for k in range(self.n_V2V):
            #                 pos1 = self.vehPos[self.V2I[m]]
            #                 pos2 = self.vehPos[self.V2VReceiver[k]]
            #                 dist_mk = np.sqrt(np.sum((pos1 - pos2) ** 2))
            #                 self.V2I_V2V_interference_channel_dB[m, k] = self.lrg_generator.generate_fading_V2V(dist_mk)
            V2I_interference_power_dB = self.power_V2I_dB + self.V2I_V2V_interference_channel_with_fast_dB[
                RB_i, i, RB_i]
            V2V_interference[i] += 10 ** (V2I_interference_power_dB / 10)
            for k in range(self.n_V2V):
                # 来自其他链路的干扰
                if not active_links[k] or i == k:
                    continue
                RB_k = RB_selection[k]
                power_dB_k = power_selection[k]
                if RB_k == RB_i:
                    power_k2i = self.power_list_V2V_dB[power_dB_k] + self.V2V_V2V_channel_with_fast_dB[k, i, RB_k]
                    V2V_interference[i] += 10 ** (power_k2i / 10)
        V2V_interference += self.sig2
        V2V_rate = np.log2(1 + np.divide(V2V_signal, V2V_interference))

        Eve_interference = np.zeros((self.n_V2V, self.n_Eve))
        Eve_signal = np.zeros((self.n_V2V, self.n_Eve))
        for i in range(self.n_V2V):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            if not active_links[i]:
                continue
            # compute receiver signal strength for current V2V link
            receiver_power_dB = self.power_list_V2V_dB[power_dB_i] + self.Eve_channel_with_fast_dB[i, :, RB_i]
            Eve_signal[i] = 10 ** (receiver_power_dB / 10)
            # 来自V2I对窃听的干扰
            V2I_Eve_interference_power_dB = self.power_V2I_dB + self.V2I_Eve_interference_channel_with_fast_dB[RB_i, :,
                                                                RB_i]
            Eve_interference[i] += 10 ** (V2I_Eve_interference_power_dB / 10)
            # 来自V2V对窃听的干扰
            for k in range(self.n_V2V):
                if not active_links[k] or i == k:
                    continue
                RB_k = RB_selection[k]
                power_dB_k = power_selection[k]
                if RB_k == RB_i:
                    power_k2i = self.power_list_V2V_dB[power_dB_k] + self.Eve_channel_with_fast_dB[k, :, RB_k]
                    Eve_interference[i] += 10 ** (power_k2i / 10)
        Eve_interference += self.sig2
        Eve_rate = np.log2(1 + np.divide(Eve_signal, Eve_interference))
        # a = np.max(Eve_rate, axis=-1)
        Secrecy_rate = V2V_rate - np.max(Eve_rate, axis=-1)
        Secrecy_rate[Secrecy_rate <= 0] = 0

        return V2I_rate, V2V_rate, Eve_rate, Secrecy_rate


    def compute_max_V2I(self):
        # compute V2I rate
        V2I_interference = np.zeros(self.n_V2I)
        V2I_interference += self.sig2
        V2I_power_dB = self.power_V2I_dB + self.V2I_channel_with_fast_dB.diagonal()
        V2I_power = 10 ** (V2I_power_dB / 10)
        V2I_rate = np.log2(1 + np.divide(V2I_power, V2I_interference))
        return V2I_rate

    def step_ddpg(self, RB, PW):
        V2I_rate, V2V_rate, Eve_rate, Secrecy_rate = self.compute_rate_ddpg(RB, PW)

        self.demand_ddpg -= Secrecy_rate * self.time_fast_fading * self.bandwidth

        # self.demand -= V2V_rate * self.time_fast_fading * self.bandwidth
        self.demand_ddpg[self.demand_ddpg <= 0] = 0
        self.individual_time_limit_ddpg -= self.time_fast_fading
        Secrecy_rate = Secrecy_rate[self.active_links_ddpg == 1]
        self.active_links_ddpg[self.demand_ddpg <= 0] = 0


        return V2I_rate, V2V_rate, Secrecy_rate

    def step_dqn(self, action):
        V2I_rate, V2V_rate, Eve_rate, Secrecy_rate = self.compute_rate(action, 'dqn')
        self.demand_dqn -= Secrecy_rate * self.time_fast_fading * self.bandwidth
        # self.demand -= V2V_rate * self.time_fast_fading * self.bandwidth
        self.demand_dqn[self.demand_dqn <= 0] = 0
        self.individual_time_limit_dqn -= self.time_fast_fading
        Secrecy_rate = Secrecy_rate[self.active_links_dqn == 1]
        self.active_links_dqn[self.demand_dqn <= 0] = 0

        return V2I_rate, V2V_rate, Secrecy_rate

    def step_ppo(self, action):
        V2I_rate, V2V_rate, Eve_rate, Secrecy_rate = self.compute_rate(action, 'ppo')
        self.demand_ppo -= Secrecy_rate * self.time_fast_fading * self.bandwidth
        # self.demand -= V2V_rate * self.time_fast_fading * self.bandwidth
        self.demand_ppo[self.demand_ppo <= 0] = 0
        self.individual_time_limit_ppo -= self.time_fast_fading
        Secrecy_rate = Secrecy_rate[self.active_links_ppo == 1]
        self.active_links_ppo[self.demand_ppo <= 0] = 0
        # self.compute_V2V_interference_ppo(action.copy())

        return V2I_rate, V2V_rate, Secrecy_rate

    def step_rand(self, action):
        V2I_rate, V2V_rate, Eve_rate, Secrecy_rate = self.compute_rate(action, 'rand')
        self.demand_rand -= Secrecy_rate * self.time_fast_fading * self.bandwidth
        self.demand_rand[self.demand_rand <= 0] = 0
        Secrecy_rate = Secrecy_rate[self.active_links_rand == 1]
        self.active_links_rand[self.demand_rand <= 0] = 0

        return V2I_rate, V2V_rate, Secrecy_rate

    def step_dpra(self, action):
        V2I_rate, V2V_rate, Eve_rate, Secrecy_rate = self.compute_rate(action, 'dpra')
        self.demand_dpra -= Secrecy_rate * self.time_fast_fading * self.bandwidth
        self.demand_dpra[self.demand_dpra <= 0] = 0
        Secrecy_rate = Secrecy_rate[self.active_links_dpra == 1]
        self.active_links_dpra[self.demand_dpra <= 0] = 0

        return V2I_rate, V2V_rate, Secrecy_rate

    def plot_dynamic_car(self):
        fig, ax = plt.subplots()
        for t in range(1000):
            self.update_env_slow()
            self.update_V2VReceiver()
            ax.set_ylim([0, self.numLane * self.laneWidth + self.disBstoHwy + self.laneWidth])
            ax.set_xlim([-self.d0, self.d0])
            for i in range(self.numLane + 1):
                if i == self.numLane / 2 or i == self.numLane or i == 0:
                    ax.plot([-self.d0, self.d0], [i * self.laneWidth + self.disBstoHwy] * 2, color='black')
                else:
                    ax.plot([-self.d0, self.d0], [i * self.laneWidth + self.disBstoHwy] * 2, linestyle='--',
                            color='black')

            ax.scatter(self.vehPos[:, 0], self.vehPos[:, 1], color='blue', marker='.')
            ax.scatter(self.vehPos[self.V2VTransmitter, 0], self.vehPos[self.V2VTransmitter, 1], color='green',
                       marker='.')
            ax.scatter(self.vehPos[self.V2I, 0], self.vehPos[self.V2I, 1], color='yellow', marker='.')
            ax.scatter(self.vehPos[self.V2VReceiver, 0], self.vehPos[self.V2VReceiver, 1], color='red', marker='.')
            ax.scatter(self.evePos[:, 0], self.evePos[:, 1], color='black')

            ax.text(1, 1, 'Time={}ms'.format(t * 100))

            plt.pause(1e-2)
            ax.cla()

        plt.show()


    def step_IHA(self):
        
        R_km = np.zeros([self.n_V2V, self.n_V2I])
        Rs_km = np.zeros([self.n_V2V, self.n_V2I]) + 100
        R_m = np.zeros([self.n_V2V, self.n_V2I])
        V2V_dB = 23
        V2I_dB = 23
        # idxex = np.argwhere(self.active_links_dpra == 1)[:,0]
        # input_matrix = np.zeros([len(idxes), self.n_V2I])
        for m in range(self.n_V2I):
            P_V2I = 10 ** ((V2I_dB + self.V2I_channel_with_fast_dB[m, m]) / 10)


            for k in range(self.n_V2V):
                if self.active_links_dpra[k] == 0:
                    R_m[k, m] = np.log2(1 + np.divide(P_V2I, self.sig2))
                    continue
                P_V2V = 10 ** ((V2V_dB + self.V2V_V2V_channel_with_fast_dB[k, k, m]) / 10)
                P_Inter = 10 ** ((V2I_dB + self.V2I_V2V_interference_channel_with_fast_dB[m, k, m]) / 10)
                P_V2I_Inter = 10 ** ((V2V_dB + self.V2V_V2I_interference_channel_with_fast_dB[k, m]) / 10)
                R = np.log2(1 + np.divide(P_V2V, self.sig2 + P_Inter))
                R_km[k, m] = R
                for e in range(self.n_Eve):
                    P_V2E = 10 ** ((V2V_dB + self.Eve_channel_with_fast_dB[k, e, m]) / 10)
                    P_V2E_Inter = 10 ** ((V2I_dB + self.V2I_Eve_interference_channel_with_fast_dB[m, e, m]) / 10)
                    Re = np.log2(1 + np.divide(P_V2E, self.sig2 + P_V2E_Inter))
                    Rs = max(0, R - Re)
                    # Rs = max(0, Rs)
                    Rs_km[k, m] = min(Rs, Rs_km[k, m])
                R_m[k, m] = np.log2(1 + np.divide(P_V2I, self.sig2 + P_V2I_Inter))
        input_matrix = Rs_km.copy()
        assignments = self.munkres.compute(-input_matrix.copy())
        # ass2 = self.munkres.compute(-R_mk)
        # print(assignments)
        assignments = np.asarray(assignments)  
        Secrecy_Rate = np.zeros([self.n_V2V])
        V2I_Rate = np.zeros([self.n_V2I])
        V2V_Rate = np.zeros([self.n_V2V])
        for row, column in assignments:
            Secrecy_Rate[row] = Rs_km[row][column]
            V2I_Rate[column] = R_m[row][column]
            V2V_Rate[row] = R_km[row][column]
        
        Secrecy_Rate[self.active_links_dpra == 0] = 0
        self.demand_dpra -= Secrecy_Rate * self.time_fast_fading * self.bandwidth
        self.demand_dpra[self.demand_dpra <= 0] = 0
        self.active_links_dpra[self.demand_dpra <= 0] = 0

        return V2I_Rate, V2V_Rate, Secrecy_Rate

    # def step_SARBPA(self):
    #
    #     R_km = np.zeros([self.n_V2V, self.n_V2I])
    #     Rs_km = np.zeros([self.n_V2V, self.n_V2I]) + 100
    #     R_m = np.zeros([self.n_V2V, self.n_V2I])
    #     V2V_dB = 23
    #     V2I_dB = 23
    #     for m in range(self.n_V2I):
    #         P_V2I = 10 ** ((V2I_dB + self.V2I_channel_with_fast_dB[m, m]) / 10)
    #
    #         for k in range(self.n_V2V):
    #             if self.active_links_dpra[k] == 0:
    #                 R_m[k, m] = np.log2(1 + np.divide(P_V2I, self.sig2))
    #                 continue
    #
    #             for e in range(self.n_Eve):
    #                 h_ks_ = self.V2I_channel_with_fast_dB[m, m]
    #                 h_m_ = self.V2V_V2V_channel_with_fast_dB[k, k, m]
    #                 g_km_ = self.V2I_V2V_interference_channel_with_fast_dB[m, k, m]
    #                 g_ms_ = self.V2V_V2I_interference_channel_with_fast_dB[k, m]
    #                 g_ke_ = self.V2I_Eve_interference_channel_with_fast_dB[m, e, m]
    #                 z_me_ = self.Eve_channel_with_fast_dB[k, e, m]
    #                 Sec_req = (self.demand_dpra[k] / self.individual_time_limit_dpra[k]) / self.bandwidth
    #                 V2I_dB, V2V_dB = self.compute_opt_power(h_ks_, h_m_, g_km_, g_ms_, g_ke_, z_me_, Qos=5, Sec_req=Sec_req)
    #                 P_V2V = 10 ** ((V2V_dB + self.V2V_V2V_channel_with_fast_dB[k, k, m]) / 10)
    #                 P_Inter = 10 ** ((V2I_dB + self.V2I_V2V_interference_channel_with_fast_dB[m, k, m]) / 10)
    #                 P_V2I_Inter = 10 ** ((V2V_dB + self.V2V_V2I_interference_channel_with_fast_dB[k, m]) / 10)
    #                 R = np.log2(1 + np.divide(P_V2V, self.sig2 + P_Inter))
    #                 R_km[k, m] = R
    #
    #                 P_V2E = 10 ** ((V2V_dB + self.Eve_channel_with_fast_dB[k, e, m]) / 10)
    #                 P_V2E_Inter = 10 ** ((V2I_dB + self.V2I_Eve_interference_channel_with_fast_dB[m, e, m]) / 10)
    #                 Re = np.log2(1 + np.divide(P_V2E, self.sig2 + P_V2E_Inter))
    #                 Rs = max(0, R - Re)
    #                 # Rs = max(0, Rs)
    #                 Rs_km[k, m] = min(Rs, Rs_km[k, m])
    #             P_V2I = 10 ** ((V2I_dB + self.V2I_channel_with_fast_dB[m, m]) / 10)
    #             R_m[k, m] = np.log2(1 + np.divide(P_V2I, self.sig2 + P_V2I_Inter))
    #
    #     input_matrix = Rs_km.copy()
    #     assignments = self.munkres.compute(-input_matrix.copy())
    #     # ass2 = self.munkres.compute(-R_mk)
    #     # print(assignments)
    #     assignments = np.asarray(assignments)
    #     Secrecy_Rate = np.zeros([self.n_V2V])
    #     V2I_Rate = np.zeros([self.n_V2I])
    #     V2V_Rate = np.zeros([self.n_V2V])
    #     for row, column in assignments:
    #         Secrecy_Rate[row] = Rs_km[row][column]
    #         V2I_Rate[column] = R_m[row][column]
    #         V2V_Rate[row] = R_km[row][column]
    #
    #     Secrecy_Rate[self.active_links_dpra == 0] = 0
    #     self.demand_dpra -= Secrecy_Rate * self.time_fast_fading * self.bandwidth
    #     self.demand_dpra[self.demand_dpra <= 0] = 0
    #     self.active_links_dpra[self.demand_dpra <= 0] = 0
    #     self.individual_time_limit_dpra -= self.time_fast_fading
    #     return V2I_Rate, V2V_Rate, Secrecy_Rate
    # def SARBPA(self):
        
    #     R_km = np.zeros([self.n_V2V, self.n_V2I])
    #     Rs_km = np.zeros([self.n_V2V, self.n_V2I])
        
    #     for k in range(self.n_V2V):
    #         if self.active_links_dpra[k] == 0:
    #             Rs_km[k] = 100
    #         for m in range(self.n_V2I):
    #             Rs_ke = np.zeros([self.n_V2V, self.n_Eve])
    #             for e in range(self.n_Eve):
    #                 h_ks_ = self.V2I_channel_with_fast_dB[m, m]
    #                 h_m_ = self.V2V_V2V_channel_with_fast_dB[k, k, m]
    #                 g_k_m = self.V2I_V2V_interference_channel_with_fast_dB[m, k, m]
    #                 g_ms_ = self.V2V_V2I_interference_channel_with_fast_dB[k, m]
    #                 g_ke_ = self.V2I_Eve_interference_channel_with_fast_dB[m, e, m]
    #                 z_me_ = self.Eve_channel_with_fast_dB[k, e, m]
                
                
                    
                    
                    
                
    # def compute_opt_power(self, h_ks_, h_m_, g_km_, g_ms_, g_ke_, z_me_, Qos=1, Sec_req=0):
    #     opt_pmk = 0
    #     opt_pk = 0
    #     alpha_s = 2 ** Sec_req - 1
    #     alpha_q = 2 ** Qos - 1
    #     beta = alpha_q*self.sig2/h_ks_
    #     gamma = alpha_q*(self.sig2+self.p_mm*g_ms_) / h_ks_
    #     # 计算定理中的A1，B1，C1
    #     A_1 = alpha_s * g_ke_ * g_km_
    #     B_1 = (alpha_s + 1)*z_me_*g_km_*self.p_mm - h_m_*g_ke_*self.p_mm + alpha_s*self.sig2*g_km_ + alpha_s*self.sig2*g_ke_
    #     C_1 = (alpha_s + 1)*z_me_*self.sig2*self.p_mm - h_m_*self.sig2*self.p_mm + alpha_s*(self.sig2**2)
    #     delta_1 = B_1**2 - 4*A_1*C_1
    #     # 计算定理中的A2，B2，C2
    #     A_2 = (alpha_s + 1)*z_me_*g_km_*h_ks_ - h_m_*g_ke_*h_ks_ + alpha_q*alpha_s*g_ms_*g_ke_*g_km_
    #     B_2 = (alpha_s + 1)*z_me_*h_ks_*self.sig2 - h_m_*h_ks_*self.sig2 - alpha_q*(alpha_s + 1)*z_me_*g_km_*self.sig2 + \
    #             alpha_q*h_m_*g_ke_*self.sig2 + alpha_q*alpha_s*g_ms_*g_km_*self.sig2 + alpha_q*alpha_s*g_ms_*g_ke_*self.sig2
    #     C_2 = alpha_q*alpha_s*g_ms_*(self.sig2**2) - alpha_q*(alpha_s + 1)*z_me_*(self.sig2**2) + alpha_q*h_m_*(self.sig2**2)
    #     delta_2 = B_2**2 - 4*A_2*C_2
    #     # 定理1中的H函数
    #     def H_(x):
    #         return (h_ks_ * x - alpha_q * self.sig2) / (alpha_q * g_ms_)
    #     # 标志功率是否还需分配
    #     flag = True
    #     case = 0
    #     # case 1
    #     if delta_1 > 0:
    #         r_11 = (-B_1 + math.sqrt(delta_1)) / (2*A_1)
    #         r_12 = (-B_1 - math.sqrt(delta_1)) / (2*A_1)
    #         if  min(r_11, self.p_km) >= max(beta, gamma, r_12):
    #             opt_pk = max(beta, gamma, r_12)
    #             opt_pmk = self.p_mm
    #             flag = False
    #
    #             # return opt_pk, opt_pmk
    #
    #     # 进入case2
    #     if delta_1 > 0 and flag:
    #         r_11 = (-B_1 + math.sqrt(delta_1)) / (2*A_1)
    #         r_12 = (-B_1 - math.sqrt(delta_1)) / (2*A_1)
    #         # case2-1
    #         if A_2 < 0 and delta_2 < 0 and min(self.p_km, gamma, r_11) >= max(beta, r_12):
    #             opt_pk = min(self.p_km, gamma, r_11)
    #             opt_pmk = H_(opt_pk)
    #             flag = False
    #
    #             # return opt_pk, opt_pmk
    #         # case2-2 and case 2-3
    #         if A_2 < 0 and delta_2 > 0 and flag:
    #             r_21 = (-B_2 + math.sqrt(delta_2)) / (2*A_2)
    #             r_22 = (-B_2 - math.sqrt(delta_2)) / (2*A_2)
    #             if min(self.p_km, gamma, r_11) >= max(beta, r_12, r_21):
    #                 opt_pk = min(self.p_km, gamma, r_11)
    #                 opt_pmk = H_(opt_pk)
    #                 flag = False
    #
    #                 # return opt_pk, opt_pmk
    #             if r_22 >= min(self.p_km, gamma, r_11) >= max(beta, r_12) and flag:
    #                 opt_pk = min(self.p_km, gamma, r_11)
    #                 opt_pmk = H_(opt_pk)
    #                 flag = False
    #
    #
    #     # 进入case 3
    #     if delta_1 > 0 and A_2 < 0 and delta_2 > 0 and flag:
    #         r_11 = (-B_1 + math.sqrt(delta_1)) / (2*A_1)
    #         r_12 = (-B_1 - math.sqrt(delta_1)) / (2*A_1)
    #         r_21 = (-B_2 + math.sqrt(delta_2)) / (2*A_2)
    #         r_22 = (-B_2 - math.sqrt(delta_2)) / (2*A_2)
    #         if r_21 >= min(self.p_km, gamma, r_11) >= r_22 >= max(beta, r_12):
    #             opt_pk = r_22
    #             opt_pmk = H_(opt_pk)
    #             flag = False
    #
    #
    #             # return opt_pk, opt_pmk
    #     # 进入case 4
    #     if delta_1 > 0 and A_2 > 0 and delta_2 > 0 and flag:
    #         r_11 = (-B_1 + math.sqrt(delta_1)) / (2*A_1)
    #         r_12 = (-B_1 - math.sqrt(delta_1)) / (2*A_1)
    #         r_21 = (-B_2 + math.sqrt(delta_2)) / (2*A_2)
    #         r_22 = (-B_2 - math.sqrt(delta_2)) / (2*A_2)
    #         if min(self.p_km, gamma, r_11, r_21) >= max(beta, r_12, r_22):
    #             opt_pk = min(self.p_km, gamma, r_11, r_21)
    #             opt_pmk = H_(opt_pk)
    #             flag = False
    #             
    #
    #
    #     if flag:
    #         opt_pk = min(self.p_km, beta)
    #         opt_pmk = 0
    #
    #     return opt_pk, opt_pmk
        
        
        
        







