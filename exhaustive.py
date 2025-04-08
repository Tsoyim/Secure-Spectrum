import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import time



def search4(env):
    n_power_level = 1
    action_dpra = np.zeros([env.n_V2V, 2], dtype='int64')
    store_action = np.zeros([(env.n_Channel * n_power_level) ** env.n_V2V, env.n_V2V])
    rate_all_dpra = []
    t = 0
    channel_list = [c for c in range(env.n_V2I)]

    for i in channel_list:
        if np.sum(env.active_links_dpra == 0) == 4:
            break
        channel_list_1 = channel_list.copy()
        channel_list_1.remove(i)
        for j in channel_list_1:
            channel_list_2 = channel_list_1.copy()
            channel_list_2.remove(j)
            for m in channel_list_2:
                channel_list_3 = channel_list_2.copy()
                channel_list_3.remove(m)
                for n in channel_list_3:
                    action_dpra[0, 0] = i
                    action_dpra[0, 1] = 0

                    action_dpra[1, 0] = j
                    action_dpra[1, 1] = 0

                    action_dpra[2, 0] = m
                    action_dpra[2, 1] = 0

                    action_dpra[3, 0] = n
                    action_dpra[3, 1] = 0

                    action_temp_findMax = action_dpra.copy()
                    V2I_rate, V2V_rate, Eve_rate, Secrecy_rate = env.compute_rate(action_temp_findMax, 'dpra')
                    rate_all_dpra.append(np.sum(Secrecy_rate))

                    store_action[t, :] = [i, j, m, n]
                    t += 1

    if len(rate_all_dpra) == 0:
        rate_all_dpra.append(1)
    i = store_action[np.argmax(rate_all_dpra), 0]
    j = store_action[np.argmax(rate_all_dpra), 1]
    m = store_action[np.argmax(rate_all_dpra), 2]
    n = store_action[np.argmax(rate_all_dpra), 3]

    action_testing_dpra = np.zeros([env.n_V2V, 2], dtype='int64')
    action_testing_dpra[0, 0] = i
    action_testing_dpra[0, 1] = 0

    action_testing_dpra[1, 0] = j
    action_testing_dpra[1, 1] = 0

    action_testing_dpra[2, 0] = m
    action_testing_dpra[2, 1] = 0

    action_testing_dpra[3, 0] = n
    action_testing_dpra[3, 1] = 0

    return action_testing_dpra


def search6(env):
    n_power_level = 1
    action_dpra = np.zeros([env.n_V2V, 2], dtype='int64')
    store_action = np.zeros([(env.n_Channel * n_power_level) ** env.n_V2V, env.n_V2V])
    rate_all_dpra = []
    t = 0
    channel_list = [c for c in range(env.n_V2I)]

    for i in channel_list:
        if np.sum(env.active_links_dpra == 0) == 6:
            break
        channel_list_1 = channel_list.copy()
        channel_list_1.remove(i)
        for j in channel_list_1:
            channel_list_2 = channel_list_1.copy()
            channel_list_2.remove(j)
            for m in channel_list_2:
                channel_list_3 = channel_list_2.copy()
                channel_list_3.remove(m)
                for n in channel_list_3:
                    channel_list_4 = channel_list_3.copy()
                    channel_list_4.remove(n)
                    for o in channel_list_4:
                        channel_list_5 = channel_list_4.copy()
                        channel_list_5.remove(o)
                        for p in channel_list_5:
                            channel_list_6 = channel_list_5.copy()
                            channel_list_6.remove(p)

                            action_dpra[0, 0] = i
                            action_dpra[0, 1] = 0

                            action_dpra[1, 0] = j
                            action_dpra[1, 1] = 0

                            action_dpra[2, 0] = m
                            action_dpra[2, 1] = 0

                            action_dpra[3, 0] = n
                            action_dpra[3, 1] = 0

                            action_dpra[4, 0] = o
                            action_dpra[4, 1] = 0

                            action_dpra[5, 0] = p
                            action_dpra[5, 1] = 0

                            action_temp_findMax = action_dpra.copy()
                            V2I_rate, V2V_rate, Eve_rate, Secrecy_rate = env.compute_rate(action_temp_findMax, 'dpra')
                            rate_all_dpra.append(np.sum(Secrecy_rate))

                            store_action[t, :] = [i, j, m, n, o, p]
                            t += 1

    if len(rate_all_dpra) == 0:
        rate_all_dpra.append(1)

    i = store_action[np.argmax(rate_all_dpra), 0]
    j = store_action[np.argmax(rate_all_dpra), 1]
    m = store_action[np.argmax(rate_all_dpra), 2]
    n = store_action[np.argmax(rate_all_dpra), 3]
    k = store_action[np.argmax(rate_all_dpra), 4]
    l = store_action[np.argmax(rate_all_dpra), 5]

    action_testing_dpra = np.zeros([env.n_V2V, 2], dtype='int64')
    action_testing_dpra[0, 0] = i % env.n_Channel
    action_testing_dpra[0, 1] = int(np.floor(i / env.n_Channel))  # power level

    action_testing_dpra[1, 0] = j % env.n_Channel
    action_testing_dpra[1, 1] = int(np.floor(j / env.n_Channel))  # power level

    action_testing_dpra[2, 0] = m % env.n_Channel
    action_testing_dpra[2, 1] = int(np.floor(m / env.n_Channel))  # power level

    action_testing_dpra[3, 0] = n % env.n_Channel
    action_testing_dpra[3, 1] = int(np.floor(n / env.n_Channel))  # power level

    action_testing_dpra[4, 0] = k % env.n_Channel
    action_testing_dpra[4, 1] = int(np.floor(k / env.n_Channel))  # power level

    action_testing_dpra[5, 0] = l % env.n_Channel
    action_testing_dpra[5, 1] = int(np.floor(l / env.n_Channel))  # power level

    return action_testing_dpra


def search8_worker(env, i, j, m, n, k, l, o, p, rate_all_dpra, store_action, t):
    # start_time = time.time()

    # 在这里放入你的代码


    action_dpra = np.zeros([env.n_V2V, 2], dtype='int64')
    action_dpra[0, 0] = i % env.n_Channel
    action_dpra[0, 1] = j % 4

    action_dpra[1, 0] = j % env.n_Channel
    action_dpra[1, 1] = m % 4

    action_dpra[2, 0] = m % env.n_Channel
    action_dpra[2, 1] = n % 4

    action_dpra[3, 0] = n % env.n_Channel
    action_dpra[3, 1] = k % 4

    action_dpra[4, 0] = k % env.n_Channel
    action_dpra[4, 1] = l % 4

    action_dpra[5, 0] = l % env.n_Channel
    action_dpra[5, 1] = i % 4

    action_dpra[6, 0] = o % env.n_Channel
    action_dpra[6, 1] = o % 4

    action_dpra[7, 0] = p % env.n_Channel
    action_dpra[7, 1] = p % 4

    action_temp_findMax = action_dpra.copy()
    V2I_rate, V2V_rate, Eve_rate, Secrecy_rate = env.compute_rate(action_temp_findMax, 'dpra')
    rate_all_dpra.append(np.sum(Secrecy_rate))

    store_action[t, :] = [i, j, m, n, k, l, o, p]

    # end_time = time.time()
    # elapsed_time = end_time - start_time

    # print("Elapsed time: {:.32f} seconds".format(elapsed_time))

def search8(env):
    n_power_level = 1
    action_dpra = np.zeros([env.n_V2V, 2], dtype='int64')
    n_agents = 8  # Now we have 8 V2V links
    store_action = np.zeros([n_agents])
    max_Sec_Rate = 0
    rate_all_dpra = []
    t = 0

    channel_list = [c for c in range(env.n_V2I)]

    for i in channel_list:
        if np.sum(env.active_links_dpra == 0) == 8:
            break
        channel_list_1 = channel_list.copy()
        channel_list_1.remove(i)
        for j in channel_list_1:
            channel_list_2 = channel_list_1.copy()
            channel_list_2.remove(j)
            for m in channel_list_2:
                channel_list_3 = channel_list_2.copy()
                channel_list_3.remove(m)
                for n in channel_list_3:
                    channel_list_4 = channel_list_3.copy()
                    channel_list_4.remove(n)
                    for o in channel_list_4:
                        channel_list_5 = channel_list_4.copy()
                        channel_list_5.remove(o)
                        for p in channel_list_5:
                            channel_list_6 = channel_list_5.copy()
                            channel_list_6.remove(p)
                            for q in channel_list_6:
                                channel_list_7 = channel_list_6.copy()
                                channel_list_7.remove(q)
                                for r in channel_list_7:
                                    channel_list_8 = channel_list_7.copy()
                                    channel_list_8.remove(r)

                                    action_dpra[0, 0] = i
                                    action_dpra[0, 1] = 0

                                    action_dpra[1, 0] = j
                                    action_dpra[1, 1] = 0

                                    action_dpra[2, 0] = m
                                    action_dpra[2, 1] = 0

                                    action_dpra[3, 0] = n
                                    action_dpra[3, 1] = 0

                                    action_dpra[4, 0] = o
                                    action_dpra[4, 1] = 0

                                    action_dpra[5, 0] = p
                                    action_dpra[5, 1] = 0

                                    action_dpra[6, 0] = q
                                    action_dpra[6, 1] = 0

                                    action_dpra[7, 0] = r
                                    action_dpra[7, 1] = 0

                                    action_temp_findMax = action_dpra.copy()
                                    V2I_rate, V2V_rate, Eve_rate, Secrecy_rate = env.compute_rate(action_temp_findMax, 'dpra')
                                    if np.sum(Secrecy_rate) > max_Sec_Rate:
                                        max_Sec_Rate = np.sum(Secrecy_rate)
                                        store_action = [i, j, m, n, o, p, q, r]
                                    t += 1



    if len(rate_all_dpra) == 0:
        rate_all_dpra.append(1)

    i = store_action[0]
    j = store_action[1]
    m = store_action[2]
    n = store_action[3]
    o = store_action[4]
    p = store_action[5]
    q = store_action[6]
    r = store_action[7]

    action_testing_dpra = np.zeros([env.n_V2V, 2], dtype='int64')
    action_testing_dpra[0, 0] = i % env.n_Channel
    action_testing_dpra[0, 1] = int(np.floor(i / env.n_Channel))  # power level

    action_testing_dpra[1, 0] = j % env.n_Channel
    action_testing_dpra[1, 1] = int(np.floor(j / env.n_Channel))  # power level

    action_testing_dpra[2, 0] = m % env.n_Channel
    action_testing_dpra[2, 1] = int(np.floor(m / env.n_Channel))  # power level

    action_testing_dpra[3, 0] = n % env.n_Channel
    action_testing_dpra[3, 1] = int(np.floor(n / env.n_Channel))  # power level

    action_testing_dpra[4, 0] = o % env.n_Channel
    action_testing_dpra[4, 1] = int(np.floor(o / env.n_Channel))  # power level

    action_testing_dpra[5, 0] = p % env.n_Channel
    action_testing_dpra[5, 1] = int(np.floor(p / env.n_Channel))  # power level

    action_testing_dpra[6, 0] = q % env.n_Channel
    action_testing_dpra[6, 1] = int(np.floor(q / env.n_Channel))  # power level

    action_testing_dpra[7, 0] = r % env.n_Channel
    action_testing_dpra[7, 1] = int(np.floor(r / env.n_Channel))  # power level

    return action_testing_dpra


def search10(env):
    n_power_level = 1
    action_dpra = np.zeros([env.n_V2V, 2], dtype='int64')
    n_agents = 8  # Now we have 8 V2V links
    store_action = np.zeros([env.n_V2V])
    max_Sec_Rate = 0
    rate_all_dpra = []
    t = 0

    channel_list = [c for c in range(env.n_V2I)]

    for i in channel_list:
        if np.sum(env.active_links_dpra == 0) == 10:
            break
        channel_list_1 = channel_list.copy()
        channel_list_1.remove(i)
        for j in channel_list_1:
            channel_list_2 = channel_list_1.copy()
            channel_list_2.remove(j)
            for m in channel_list_2:
                channel_list_3 = channel_list_2.copy()
                channel_list_3.remove(m)
                for n in channel_list_3:
                    channel_list_4 = channel_list_3.copy()
                    channel_list_4.remove(n)
                    for o in channel_list_4:
                        channel_list_5 = channel_list_4.copy()
                        channel_list_5.remove(o)
                        for p in channel_list_5:
                            channel_list_6 = channel_list_5.copy()
                            channel_list_6.remove(p)
                            for q in channel_list_6:
                                channel_list_7 = channel_list_6.copy()
                                channel_list_7.remove(q)
                                for r in channel_list_7:
                                    channel_list_8 = channel_list_7.copy()
                                    channel_list_8.remove(r)
                                    for s in channel_list_8:
                                        channel_list_9 = channel_list_8.copy()
                                        channel_list_9.remove(s)
                                        for t in channel_list_9:
                                            channel_list_10 = channel_list_9.copy()
                                            channel_list_10.remove(t)

                                            action_dpra[0, 0] = i
                                            action_dpra[0, 1] = 0

                                            action_dpra[1, 0] = j
                                            action_dpra[1, 1] = 0

                                            action_dpra[2, 0] = m
                                            action_dpra[2, 1] = 0

                                            action_dpra[3, 0] = n
                                            action_dpra[3, 1] = 0

                                            action_dpra[4, 0] = o
                                            action_dpra[4, 1] = 0

                                            action_dpra[5, 0] = p
                                            action_dpra[5, 1] = 0

                                            action_dpra[6, 0] = q
                                            action_dpra[6, 1] = 0

                                            action_dpra[7, 0] = r
                                            action_dpra[7, 1] = 0

                                            action_dpra[8, 0] = s
                                            action_dpra[8, 1] = 0

                                            action_dpra[9, 0] = t
                                            action_dpra[9, 1] = 0


                                            action_temp_findMax = action_dpra.copy()
                                            V2I_rate, V2V_rate, Eve_rate, Secrecy_rate = env.compute_rate(action_temp_findMax, 'dpra')
                                            if np.sum(Secrecy_rate) > max_Sec_Rate:
                                                max_Sec_Rate = np.sum(Secrecy_rate)
                                                store_action = [i, j, m, n, o, p, q, r, s, t]
                                            

    if len(rate_all_dpra) == 0:
        rate_all_dpra.append(1)

    i = store_action[0]
    j = store_action[1]
    m = store_action[2]
    n = store_action[3]
    k = store_action[4]
    l = store_action[5]
    o = store_action[6]
    p = store_action[7]
    q = store_action[8]
    r = store_action[9]

    action_testing_dpra = np.zeros([env.n_V2V, 2], dtype='int64')
    action_testing_dpra[0, 0] = i % env.n_Channel
    action_testing_dpra[0, 1] = int(np.floor(i / env.n_Channel))  # power level

    action_testing_dpra[1, 0] = j % env.n_Channel
    action_testing_dpra[1, 1] = int(np.floor(j / env.n_Channel))  # power level

    action_testing_dpra[2, 0] = m % env.n_Channel
    action_testing_dpra[2, 1] = int(np.floor(m / env.n_Channel))  # power level

    action_testing_dpra[3, 0] = n % env.n_Channel
    action_testing_dpra[3, 1] = int(np.floor(n / env.n_Channel))  # power level

    action_testing_dpra[4, 0] = k % env.n_Channel
    action_testing_dpra[4, 1] = int(np.floor(k / env.n_Channel))  # power level

    action_testing_dpra[5, 0] = l % env.n_Channel
    action_testing_dpra[5, 1] = int(np.floor(l / env.n_Channel))  # power level

    action_testing_dpra[6, 0] = o % env.n_Channel
    action_testing_dpra[6, 1] = int(np.floor(o / env.n_Channel))  # power level

    action_testing_dpra[7, 0] = p % env.n_Channel
    action_testing_dpra[7, 1] = int(np.floor(p / env.n_Channel))  # power level

    action_testing_dpra[8, 0] = q % env.n_Channel
    action_testing_dpra[8, 1] = int(np.floor(q / env.n_Channel))  # power level

    action_testing_dpra[9, 0] = r % env.n_Channel
    action_testing_dpra[9, 1] = int(np.floor(r / env.n_Channel))  # power level

    return action_testing_dpra


def search_optimized(env):
    n_power_level = 1
    n_channels_power = env.n_Channel * n_power_level
    action_dpra = np.zeros([env.n_V2V, 2], dtype='int64')
    store_action = np.zeros([n_channels_power**env.n_V2V, env.n_V2V])
    rate_all_dpra = []

    channel_indices = np.arange(n_channels_power)

    for _ in range(env.n_V2V):
        active_links_mask = env.active_links_dpra == 0
        valid_channel_indices = channel_indices[~active_links_mask]
        permutations = np.meshgrid(*([valid_channel_indices] * env.n_V2V))
        perm_indices = np.array([p.flatten() for p in permutations]).T

        for perm_index in perm_indices:
            action_dpra[:, 0] = perm_index % env.n_Channel
            action_dpra[:, 1] = np.floor(perm_index / env.n_Channel).astype(int)

            action_temp_findMax = action_dpra.copy()
            _, _, _, Secrecy_rate = env.compute_rate(action_temp_findMax, 'dpra')
            rate_all_dpra.append(np.sum(Secrecy_rate))

            store_action[len(rate_all_dpra) - 1, :] = perm_index

    if not rate_all_dpra:
        rate_all_dpra.append(1)

    best_perm_index = store_action[np.argmax(rate_all_dpra)]
    action_testing_dpra = np.column_stack([
        best_perm_index % env.n_Channel,
        (best_perm_index // env.n_Channel).astype(int)
    ])

    return action_testing_dpra