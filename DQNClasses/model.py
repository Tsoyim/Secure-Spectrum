import torch
from torch import nn
from torch.nn import functional as F


from torch.distributions import Categorical


HIDDEN_DIM_CNN = 32
torch.cuda.manual_seed(3407)
torch.manual_seed(3407)
class DQN(nn.Module):
    def __init__(self, n_state, n_action, muti_dim=1):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_state, 512*muti_dim)
        self.fc2 = nn.Linear(512*muti_dim, 256*muti_dim)
        self.fc3 = nn.Linear(256*muti_dim, 128*muti_dim)
        self.fc4 = nn.Linear(128*muti_dim, n_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DQNCNN(nn.Module):
    def __init__(self, input_dim, n_action, kernel=5, pad=2, pool_kernel=2, stride=1):
        super(DQNCNN, self).__init__()
        HIDDEN_DIM_CNN = n_action
        self.conv1 = nn.Conv1d(1, HIDDEN_DIM_CNN, kernel_size=kernel, padding=pad, stride=stride)
        self.pool1 = nn.AvgPool1d(kernel_size=pool_kernel)
        # self.conv2 = nn.Conv1d(HIDDEN_DIM_CNN, HIDDEN_DIM_CNN * 2, kernel_size=3, padding=1)
        # self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        # 计算输出维度
        dim = ((input_dim - kernel + 2 * pad) // stride + 1) * HIDDEN_DIM_CNN // pool_kernel
        self.fc1 = nn.Linear(dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_action)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNNLayer(nn.Module):
    def __init__(self, input_dim, n_action):
        super(CNNLayer, self).__init__()
        self.conv1 = nn.Conv1d(1, HIDDEN_DIM_CNN, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(HIDDEN_DIM_CNN * input_dim // 2, 256)
        self.fc2 = nn.Linear(256, n_action)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x