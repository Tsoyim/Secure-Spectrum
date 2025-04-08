import numpy as np
import matplotlib.pyplot as plt

# 读取NumPy数据文件
# file_path = 'experiment_result/MADQN_no/train_result/reward_V2I_4_V2V_4_Eve_1_discount0.95.npy'
file_path = 'experiment_result/MADDPG_no/train_result/reward_V2I_4_V2V_4_Eve_1.npy'
reward_data = np.load(file_path)

# 计算滑动平均
window_size = 5# 滑动窗口大小
reward_smooth = np.convolve(reward_data, np.ones(window_size) / window_size, mode='valid')

# 绘制原始数据和滑动平均
# plt.figure(figsize=(10, 6))
# plt.plot(reward_data, label='Original Data')
plt.plot(reward_smooth, label=f'Reward')
plt.xlabel('episode')
plt.ylabel('Reward')
plt.legend()
# plt.title('Returns')
plt.grid(True)
plt.show()

# import numpy as np
# import torch
# from torch.utils.tensorboard import SummaryWriter
#
# # 读取NumPy数据文件
# file_path = './experiment_result/MADQN_no/train_result/V2I_4_V2V_4_Eve_1_fit_result.npy'
# reward_data = np.load(file_path)
#
# # 创建一个TensorBoard记录器
# writer = SummaryWriter()
#
# # 将 NumPy 数组的数据写入 TensorBoard
# # 注意：PyTorch的SummaryWriter的add_scalar接受三个参数，分别是标签、数据、步数
# for i, value in enumerate(reward_data):
#     writer.add_scalar('reward_data', value, i)
#
# # 关闭记录器
# writer.close()
