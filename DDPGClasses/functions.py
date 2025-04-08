
import torch
import torch.nn.functional as F
def get_one_hot(prob_distribution):
    # 找到最大概率的索引
    max_prob_index = torch.argmax(prob_distribution, dim=-1)

    # 将最大概率的索引转换为 One-Hot 编码
    one_hot_encoding = F.one_hot(max_prob_index, num_classes=prob_distribution.size(-1))

    return one_hot_encoding