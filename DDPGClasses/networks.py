import torch
from torch import nn
from torch.nn import functional as F
import torch as T
import sys

from torch.distributions import Categorical

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

torch.cuda.manual_seed(3407)
torch.manual_seed(3407)
class ActorNetwork(nn.Module):
    def __init__(self, n_state, n_action_ch, n_action_pw, actor_mutli=1):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(n_state, int(actor_mutli*512))
        self.fc2 = nn.Linear(int(actor_mutli*512), int(actor_mutli*256))
        self.fc3 = nn.Linear(int(actor_mutli*256), int(actor_mutli*128))
        self.ch_out = nn.Linear(int(actor_mutli*128), n_action_ch)
        self.pw_out = nn.Linear(int(actor_mutli*128), n_action_pw)
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)
        # orthogonal_init(self.fc4)
        init_w = 3e-3
        # self.ch_out.weight.data.uniform_(-init_w, init_w)
        # self.ch_out.bias.data.uniform_(-init_w, init_w)
        # self.pw_out.weight.data.uniform_(-init_w, init_w)
        # self.pw_out.bias.data.uniform_(-init_w, init_w)
    def forward(self, x, train=True):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        # ch = self.ch_out(x3)
        # if train == True:
        #     ch = self.gumbel_softmax(ch, hard=True)
        ch = F.tanh(self.ch_out(x3))

        pw = F.tanh(self.pw_out(x3))


        return ch, pw

    def gumbel_softmax(self, logits, temperature=1.0, epsilon=1e-20, hard=False):
        # 从 Gumbel 分布中采样噪声
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + epsilon) + epsilon)

        # 添加 Gumbel 噪声到 logits，得到带有噪声的采样
        gumbel_logits = (logits + gumbel_noise) / temperature

        # 使用 softmax 获得概率分布
        gumbel_softmax = F.softmax(gumbel_logits, dim=-1)

        # 如果硬采样为 True，则使用 argmax 得到独热编码
        if hard:
            _, max_indices = torch.max(gumbel_softmax, dim=-1, keepdim=True)
            one_hot = torch.zeros_like(gumbel_softmax)
            one_hot.scatter_(dim=-1, index=max_indices, value=1.0)
            gumbel_softmax = (one_hot - gumbel_softmax).detach() + gumbel_softmax

        return gumbel_softmax


class CriticNetwork(nn.Module):
    def __init__(self, n_state, n_action, critic_mutli=1, with_attention=0):
        super(CriticNetwork, self).__init__()
        self.with_attention = with_attention
        if with_attention == 1:
            self.net = CriticNetworkAttention(n_state, n_action, num_heads=4)
        else:
            self.net = CriticNetworkMLP(n_state, n_action, critic_mutli)



    def forward(self, state, action):

        value = self.net.forward(state, action)

        return value




class CriticNetworkMLP(nn.Module):
    def __init__(self, n_state, n_action, critic_mutli=1):
        super(CriticNetworkMLP, self).__init__()
        self.fc1_action = nn.Linear(n_action, int(32 * critic_mutli))
        self.fc1_state = nn.Linear(n_state, int(critic_mutli*1024))
        self.fc2 = nn.Linear(int(32 * critic_mutli) + int(critic_mutli*1024), int(critic_mutli*512))
        self.fc3 = nn.Linear(int(critic_mutli*512), int(critic_mutli*256))
        self.fc4 = nn.Linear(int(critic_mutli*256), 1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)
        orthogonal_init(self.fc4)
        # self.fc5 = nn.Linear(128, 1)
        init_w = 3e-3
        # self.fc4.weight.data.uniform_(-init_w, init_w)
        # self.fc4.bias.data.uniform_(-init_w, init_w)


    def forward(self, state, action):
        state = self.fc1_state(state)
        action = self.fc1_action(action)
        x1 = F.relu(self.fc2(T.cat([state, action], dim=1)))
        x2 = F.relu(self.fc3(x1))
        value = self.fc4(x2)

        return value



class CriticNetworkAttention(nn.Module):
    def __init__(self, n_state, n_action, num_heads=4):
        super(CriticNetworkAttention, self).__init__()

        # Assuming n_state and n_action are the same for simplicity
        input_dim = n_state + n_action
        self.fc = nn.Linear(input_dim, 512)

        # Multihead Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=num_heads)

        # Fully connected layers after attention
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, state, action):
        # Concatenate state and action
        input_data = T.cat([state, action], dim=1)
        # input_data = input_data.permute(1, 0, 2)
        input_data = self.fc(input_data)
        # Apply multihead attention
        x, _ = self.attention(input_data, input_data, input_data)
        # x = x.permute(1, 0, 2)
        # Fully connected layers
        x1 = F.relu(self.fc1(x))
        value = self.fc2(x1)

        return value
