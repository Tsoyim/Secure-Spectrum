import numpy as np
from numpy.random import normal
from HighwayChannel import HwyChannelLargeScaleFadingGenerator, HwyChannelLargeScaleFadingGenerator2
import matplotlib.pyplot as plt

np.random.seed(1234)


class RLHighWayEnvironment():
    '''
    This class implement the simulator environment for reinforcement learning in the Highway scenario
    '''

    def __init__(self, config):
        '''
        Construction methods for class RLHighWayEnvironment
        :param config: dict containing key parameters for simulation
        '''

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
        # self.time_slow_fading = 0.1
        self.bandwidth = int(1e6)
        self.demand_size = config["demand"]
        self.speed = 100 / 3.6
        self.Eveknow = config["Eveknow"]
        self.n_power_level = len(self.power_list_V2V_dB)
        self.disBstoHwy = 35
        self.lambdda = 0.1
        self.POSSION = config["POSSION"]
        self.EveDist = config["EveDist"]
        # initialize vehicle and channel sampler
        self.lrg_generator = HwyChannelLargeScaleFadingGenerator()
        # internal simulation parameters
        # self.active_links = np.ones(self.numDUE, dtype="bool")
        # self.individual_time_limit = self.time_slow_fading * np.ones(self.numDUE)
        # self.demand = self.demand_size * np.ones(self.numDUE)
        self.d0 = np.sqrt(500 ** 2 - self.disBstoHwy ** 2)
        self.V2V_interference = np.zeros((self.n_V2V, self.n_V2I))
        self.V2E_interference = np.zeros((self.n_V2V, self.n_Eve, self.n_V2I))
        self.BS_position = [0, 0]
        self.numLane = 6
        self.laneWidth = 4
        self.time_block_limit = 10
        self.time_slow_fading = self.time_fast_fading * self.time_block_limit
        self.time_limit = self.time_block_limit / 1000

    def init_simulation(self):
        '''
        Initialize Highway Environment simulator
        :return:
        '''

        self.generate_vehicles_woPossion()

        # self.generate_vehicles()
        self.update_V2VReceiver()
        self.update_channels_slow()
        self.update_channels_fast()
        self.active_links = np.ones(self.n_V2V, dtype="bool")
        self.individual_time_limit = self.time_limit * np.ones(self.n_V2V)
        self.demand = self.demand_size * np.ones(self.n_V2V)

        self.active_links_rand = np.ones(self.n_V2V, dtype="bool")
        self.individual_time_limit_rand = self.time_limit * np.ones(self.n_V2V)
        self.demand_rand = self.demand_size * np.ones(self.n_V2V)

    def update_env_slow(self, POSITION=True):
        if POSITION == True:
            self.update_vehicle_position()
            self.update_V2VReceiver()
            self.update_channels_slow()
            self.update_channels_fast()
        self.active_links = np.ones(self.n_V2V, dtype="bool")
        self.individual_time_limit = self.time_limit * np.ones(self.n_V2V)
        self.demand = self.demand_size * np.ones(self.n_V2V)

        self.active_links_rand = np.ones(self.n_V2V, dtype="bool")
        self.individual_time_limit_rand = self.time_limit * np.ones(self.n_V2V)
        self.demand_rand = self.demand_size * np.ones(self.n_V2V)

    def update_env_fast(self):
        self.update_channels_fast()



    def generate_vehicles(self):
        d_avg = 2.5 * 100 / 3.6
        while 1:
            vehPos = []
            vehDir = []

            for ilane in range(self.numLane):
                nveh = int(2*self.d0 / d_avg)
                posi_ilane = np.zeros((nveh, 2))
                posi_ilane[:, 0] = (2 * np.random.rand(nveh) - 1) * self.d0
                posi_ilane[:, 1] = self.disBstoHwy + (ilane * (self.laneWidth) + self.laneWidth / 2) * np.ones(nveh)
                vehPos.append(posi_ilane)
                if ilane < self.numLane // 2:
                    vehDir.append([0] * nveh)  # left -> right
                else:
                    vehDir.append([1] * nveh)  # right -> left
            vehPos = np.concatenate(vehPos)
            vehDir = np.concatenate(vehDir)
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
        for i in range(self.n_Eve):
            posEve[i, 0] = 0
            posEve[i, 1] = self.disBstoHwy - self.EveDist
        evePos.append(posEve)
        evePos = np.concatenate(evePos)
        self.V2I = np.array(indCUE)
        self.V2VTransmitter = indDUETransmitter
        self.V2VReceiver = indDUEReceiver
        self.evePos = evePos
        self.vehPos = vehPos
        self.vehDir = vehDir

    def update_vehicle_position(self):
        '''
        Update the position of each vehicle according to their current position, direction and speed
        :return:
        '''
        factor = 1.0

        for veh_idx in range(len(self.vehPos)):
            cur_posi = self.vehPos[veh_idx]
            if self.vehDir[veh_idx] == 0:  # left -> right
                tmp = cur_posi[0] + self.speed * factor * self.time_fast_fading * self.time_block_limit
                if tmp > self.d0:
                    self.vehPos[veh_idx][0] = tmp - 2 * self.d0
                else:
                    self.vehPos[veh_idx][0] = tmp
            else:
                tmp = cur_posi[0] + self.speed * factor * self.time_fast_fading * self.time_block_limit
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
        '''
        Compute larger scale fading for five channels:
        V2I_channel_dB: channel for V2I->BS (M * 1)
        V2V_channel_dB: channel between V2V transceiver (K * 1)
        V2V_V2I_interference_channel_dB: channel between V2V transmitter and BS receiver
        V2I_V2V_interference_channel_dB: channel between V2I transmitter and V2V receiver
        V2V_V2V_interference_channel_dB: channel of V2V transmitter and receiver between different V2V pairs
        M : number of CUE, K: number of DUE
        :return:
        '''
        self.V2I_channel_dB = np.zeros(self.n_V2I)
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

        V2V_V2V_channel_with_fast_dB = np.repeat(self.V2V_V2V_channel_dB[:, :, np.newaxis], self.n_Channel, axis=2)
        fast_componet = np.abs(normal(0, 1, V2V_V2V_channel_with_fast_dB.shape) + 1j * normal(0, 1,
                                                                                              V2V_V2V_channel_with_fast_dB.shape)) / np.sqrt(
            2)
        self.V2V_V2V_channel_with_fast_dB = V2V_V2V_channel_with_fast_dB + 20 * np.log10(fast_componet)

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

    def get_state(self, idx=0, epsilon=0.02, ind_episode=1., mode='gain'):
        CSI_fast_state = []
        CSI_slow_state = []
        CSI_gain_state = []
        state = []
        # 收到的干扰
        V2V_interference = self.V2V_interference[idx, :]
        V2V_interference = (-V2V_interference - 60) / 60
        load_remaining = [self.demand[idx] / self.demand_size]
        time_remaining = [self.individual_time_limit[idx] / (self.time_block_limit / 1000)]

        payload_state = np.concatenate([
            np.array(load_remaining),  # 将 Python 列表转换为数组
            np.array(time_remaining),
            V2V_interference  # 将 Python 列表转换为数组
        ])

        if mode == 'gain':
            # 信道增益
            # 1 x m
            V2V_V2I_gain = self.V2V_V2I_interference_channel_with_fast_dB[idx, :]
            V2V_V2I_gain = (V2V_V2I_gain + 80) / 60
            # m x m
            V2I_V2V_gain = self.V2I_V2V_interference_channel_with_fast_dB[:, idx, :]
            V2I_V2V_gain = (V2I_V2V_gain + 80) / 60
            # k x m
            V2V_V2V_gain = self.V2V_V2V_channel_with_fast_dB[:, idx, :]
            V2V_V2V_gain = (V2V_V2V_gain + 80) / 60
            # m x m
            V2I_Eve_gain = self.V2I_Eve_interference_channel_with_fast_dB
            V2I_Eve_gain = (V2I_Eve_gain + 80) / 60
            # 1 x m

            Eve_gain = self.Eve_channel_with_fast_dB[idx]
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
            V2V_V2I_fast = self.V2V_V2I_interference_channel_with_fast_dB[idx, :] - self.V2V_V2I_interference_channel_dB[
                idx]  # h_kb
            V2I_V2V_fast = self.V2I_V2V_interference_channel_with_fast_dB[:, idx] - self.V2I_V2V_interference_channel_dB[:,
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
            state.append(self.get_state(i))
        if np.any(np.isnan(state)):
            print(1)
        state = np.concatenate(state)
        return state

    def plot_dynamic_car(self, T=1):
        fig, ax = plt.subplots()
        for t in range(T):
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

    def compute_V2V_interference(self, action):

        RB_selection = action[:, 0]
        power_selection = action[:, 1]
        # interference from V2I link
        V2V_interference = np.zeros((self.n_V2V, self.n_V2I)) + self.sig2
        for i in range(self.n_V2V):
            # if current V2V link has finished its transmission, it then turns to silent mode
            if not self.active_links[i]:
                continue
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            V2I_power_dB = self.power_V2I_dB + self.V2I_V2V_interference_channel_with_fast_dB[RB_i, i, RB_i]
            V2V_interference[i, RB_i] += 10 ** (V2I_power_dB / 10)
            # interference from other V2V link with the same RB
            for j in range(self.n_V2V):
                RB_j = RB_selection[j]
                power_dB_j = power_selection[j]
                if not self.active_links[j] or j == i:
                    continue
                if RB_j == RB_i:
                    power_j2i = self.power_list_V2V_dB[power_dB_j] + self.V2V_V2V_channel_with_fast_dB[j, i, RB_j]
                    V2V_interference[i, RB_j] += 10 ** (power_j2i / 10)
        # transform inference from linear scale to dB
        self.V2V_interference = 10 * np.log10(V2V_interference)

        V2E_interference = np.zeros((self.n_V2V, self.n_Eve, self.n_V2I)) + self.sig2
        for i in range(self.n_V2V):
            # if current V2V link has finished its transmission, it then turns to silent mode
            if not self.active_links[i]:
                continue
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            V2I_power_dB = self.power_V2I_dB + self.V2I_Eve_interference_channel_with_fast_dB[RB_i, :, RB_i]
            V2E_interference[i, :, RB_i] += 10 ** (V2I_power_dB / 10)
            # interference from other V2V link with the same RB
            for j in range(self.n_V2V):
                RB_j = RB_selection[j]
                power_dB_j = power_selection[j]
                if not self.active_links[j] or j == i:
                    continue
                if RB_j == RB_i:
                    power_j2i = self.power_list_V2V_dB[power_dB_j] + self.Eve_channel_with_fast_dB[j, :, RB_j]
                    V2E_interference[i, :, RB_j] += 10 ** (power_j2i / 10)
        # transform inference from linear scale to dB
        self.V2E_interference = 10 * np.log10(V2E_interference)

    def compute_V2V_interference_conti(self, RB, PW):

        RB_selection = RB
        power_selection = PW
        # interference from V2I link
        V2V_interference = np.zeros((self.n_V2V, self.n_V2I)) + self.sig2
        for i in range(self.n_V2V):
            # if current V2V link has finished its transmission, it then turns to silent mode
            if not self.active_links[i]:
                continue
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            V2I_power_dB = self.power_V2I_dB + self.V2I_V2V_interference_channel_with_fast_dB[RB_i, i, RB_i]
            V2V_interference[i, RB_i] += 10 ** (V2I_power_dB / 10)
            # interference from other V2V link with the same RB
            for j in range(self.n_V2V):
                RB_j = RB_selection[j]
                power_dB_j = power_selection[j]
                if not self.active_links[j] or j == i:
                    continue
                if RB_j == RB_i:
                    power_j2i = power_dB_j + self.V2V_V2V_channel_with_fast_dB[j, i, RB_j]
                    V2V_interference[i, RB_j] += 10 ** (power_j2i / 10)
        # transform inference from linear scale to dB
        self.V2V_interference = 10 * np.log10(V2V_interference)

        V2E_interference = np.zeros((self.n_V2V, self.n_Eve, self.n_V2I)) + self.sig2
        for i in range(self.n_V2V):
            # if current V2V link has finished its transmission, it then turns to silent mode
            if not self.active_links[i]:
                continue
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            V2I_power_dB = self.power_V2I_dB + self.V2I_Eve_interference_channel_with_fast_dB[RB_i, :, RB_i]
            V2E_interference[i, :, RB_i] += 10 ** (V2I_power_dB / 10)
            # interference from other V2V link with the same RB
            for j in range(self.n_V2V):
                RB_j = RB_selection[j]
                power_dB_j = power_selection[j]
                if not self.active_links[j] or j == i:
                    continue
                if RB_j == RB_i:
                    power_j2i = power_dB_j + self.Eve_channel_with_fast_dB[j, :, RB_j]
                    V2E_interference[i, :, RB_j] += 10 ** (power_j2i / 10)
        # transform inference from linear scale to dB
        self.V2E_interference = 10 * np.log10(V2E_interference)


    def compute_rate(self, action):


        RB_selection = action[:, 0]
        power_selection = action[:, 1]
        # ----------------compute V2I rate-----------------------
        # check_ok
        V2I_interference = np.zeros(self.n_V2I)
        for i in range(self.n_V2V):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            # 链路如果没被激活,则不需要计算干扰
            if not self.active_links[i]:
                continue
            # 来自V2V的干扰
            # pkd*gkm_____ok
            interference = self.power_list_V2V_dB[power_dB_i] + self.V2V_V2I_interference_channel_with_fast_dB[i, RB_i]
            # 转化为线性增益
            V2I_interference[RB_i] += 10 ** (interference / 10)
        # 加环境噪声
        V2I_interference += self.sig2
        # V2I的信号强度,pmc*gm_____ok
        V2I_power_dB = self.power_V2I_dB + self.V2I_channel_with_fast_dB.diagonal()
        V2I_power = 10 ** (V2I_power_dB / 10)
        # Shannon定理
        V2I_rate = np.log2(1 + np.divide(V2I_power, V2I_interference))

        # ----------------compute V2V rate-----------------------
        V2V_interference = np.zeros(self.n_V2V)
        V2V_signal = np.zeros(self.n_V2V)
        for i in range(self.n_V2V):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            if not self.active_links[i]:
                continue
            # compute receiver signal strength for current V2V link,pkd*gk
            receiver_power_dB = self.power_list_V2V_dB[power_dB_i] + self.V2V_V2V_channel_with_fast_dB[i, i, RB_i]
            V2V_signal[i] = 10 ** (receiver_power_dB / 10)
            # compute V2I link interference to current V2V link, pmc*gmk
            V2I_interference_power_dB = self.power_V2I_dB + self.V2I_V2V_interference_channel_with_fast_dB[
                RB_i, i, RB_i]
            # 转化为线性增益
            V2V_interference[i] += 10 ** (V2I_interference_power_dB / 10)
            for k in range(self.n_V2V):
                # 来自其他链路的干扰
                if not self.active_links[k] or i == k:
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
            if not self.active_links[i]:
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
                if not self.active_links[k] or i == k:
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
        # print(self.V2I_channel_with_fast_dB.diagonal())
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

    def compute_rate_conti(self, RB, PW):

        RB_selection = RB
        power_selection = PW
        # compute V2I rate
        V2I_interference = np.zeros(self.n_V2I)
        for i in range(self.n_V2V):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            if not self.active_links[i]:
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
            if not self.active_links[i]:
                continue
            # compute receiver signal strength for current V2V link
            receiver_power_dB = power_dB_i + self.V2V_V2V_channel_with_fast_dB[i, i, RB_i]
            V2V_signal[i] = 10 ** (receiver_power_dB / 10)
            # compute V2I link interference to current V2V link
            V2I_interference_power_dB = self.power_V2I_dB + self.V2I_V2V_interference_channel_with_fast_dB[
                RB_i, i, RB_i]
            V2V_interference[i] += 10 ** (V2I_interference_power_dB / 10)
            for k in range(self.n_V2V):
                if not self.active_links[k] or i == k:
                    continue
                RB_k = RB_selection[k]
                power_dB_k = power_selection[k]
                if RB_k == RB_i:
                    # k对i的干扰
                    power_k2i = power_dB_k + self.V2V_V2V_channel_with_fast_dB[k, i, RB_k]
                    V2V_interference[i] += 10 ** (power_k2i / 10)
        V2V_interference += self.sig2
        V2V_rate = np.log2(1 + np.divide(V2V_signal, V2V_interference))

        Eve_interference = np.zeros((self.n_V2V, self.n_Eve))
        Eve_signal = np.zeros((self.n_V2V, self.n_Eve))
        for i in range(self.n_V2V):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            if not self.active_links[i]:
                continue
            # compute receiver signal strength for current V2V link
            receiver_power_dB = power_dB_i + self.Eve_channel_with_fast_dB[i, :, RB_i]
            Eve_signal[i] = 10 ** (receiver_power_dB / 10)

            V2I_Eve_interference_power_dB = self.power_V2I_dB + self.V2I_Eve_interference_channel_with_fast_dB[RB_i, :,
                                                                RB_i]
            Eve_interference[i] += 10 ** (V2I_Eve_interference_power_dB / 10)
            for k in range(self.n_V2V):
                if not self.active_links[k] or i == k:
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


    def step(self, action):
        V2I_rate, V2V_rate, Eve_rate, Secrecy_rate = self.compute_rate(action)

        self.demand -= Secrecy_rate * self.time_fast_fading * self.bandwidth

        # self.demand -= V2V_rate * self.time_fast_fading * self.bandwidth
        self.demand[self.demand <= 0] = 0
        self.individual_time_limit -= self.time_fast_fading
        remain_rate = Secrecy_rate * self.individual_time_limit * self.bandwidth - self.demand
        # reward_V2V = V2V_rate/10
        # 奖励函数设计
        reward_V2V = Secrecy_rate / 15
        reward_V2V[self.demand == 0] = 1
        Secrecy_rate = Secrecy_rate[self.active_links == 1]
        self.active_links[self.demand <= 0] = 0
        # compute combined reward
        factor = 1.0
        reward = 0.008 * np.sum(V2I_rate) / (10 * self.n_V2I) + 1 * np.sum(reward_V2V) / self.n_V2V + self.F(remain_rate)

        self.update_env_fast()
        self.compute_V2V_interference(action)


        return V2I_rate, V2V_rate, Secrecy_rate, reward

    def step_conti(self, RB, PW):
        V2I_rate, V2V_rate, Eve_rate, Secrecy_rate = self.compute_rate_conti(RB, PW)

        self.demand -= Secrecy_rate * self.time_fast_fading * self.bandwidth

        # self.demand -= V2V_rate * self.time_fast_fading * self.bandwidth
        self.demand[self.demand <= 0] = 0
        self.individual_time_limit -= self.time_fast_fading
        remain_rate = Secrecy_rate * self.individual_time_limit * self.bandwidth - self.demand
        # reward_V2V = V2V_rate/10
        # 奖励函数设计
        reward_V2V = Secrecy_rate / 15
        reward_V2V[self.demand == 0] = 1
        Secrecy_rate = Secrecy_rate[self.active_links == 1]
        self.active_links[self.demand <= 0] = 0
        # compute combined reward
        factor = 1.0
        reward_V2I = 0.01 * np.sum(V2I_rate) / (10 * self.n_V2I)
        v2v = np.sum(reward_V2V) / self.n_V2V
        reward = 0.008 * np.sum(V2I_rate) / (10 * self.n_V2I) + 1 * np.sum(reward_V2V) / self.n_V2V + self.F(remain_rate)



        self.update_env_fast()
        self.compute_V2V_interference_conti(RB, PW)


        return V2I_rate, V2V_rate, Secrecy_rate, reward


    # def step_rand(self, action):
    #
    #     V2I_rate, V2V_rate, Eve_rate, Secrecy_rate = self.compute_rate_rand(action)
    #
    #     self.demand_rand -= Secrecy_rate * self.time_fast_fading * self.bandwidth
    #     self.demand_rand[self.demand_rand <= 0] = 0
    #     self.individual_time_limit_rand -= self.time_fast_fading
    #     remain_rate = Secrecy_rate * self.individual_time_limit_rand * self.bandwidth - self.demand_rand
    #     # 奖励函数设计
    #     reward_V2V = Secrecy_rate / 15
    #     reward_V2V[self.demand_rand == 0] = 1
    #     Secrecy_rate = Secrecy_rate[self.active_links_rand == 1]
    #     self.active_links_rand[self.demand_rand <= 0] = 0
    #     # compute combined reward
    #     factor = 1.0
    #     reward = 0.2 * np.sum(V2I_rate) / (10 * self.n_V2I) + 0.8 * self.F(remain_rate) / self.n_V2V
    #
    #     # self.update_env_fast()
    #     # self.compute_V2V_interference(action)
    #
    #     return V2I_rate, V2V_rate, Secrecy_rate, reward





    def F(self, x_arr):
        for i, item in enumerate(x_arr):
            if item >= 0:
                x_arr[i] = 0.1
            else:
                x_arr[i] = 0
        return np.sum(x_arr)



