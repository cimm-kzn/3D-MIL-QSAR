import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class MBSplitter(Dataset):
    def __init__(self, x, y, m):
        super(MBSplitter, self).__init__()
        self.x = x
        self.y = y
        self.m = m

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.m[i]

    def __len__(self):
        return len(self.y)


class Swish(nn.Module):
    def forward(self, input_tensor):
        return input_tensor * torch.sigmoid(input_tensor)


class MarginLoss(nn.Module):
    def __init__(self, m_pos=0.9, m_neg=0.1, alpha=0.5):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.alpha = alpha

    def forward(self, lengths, labels):
        left = F.relu(self.m_pos - lengths, inplace=True) ** 2
        right = F.relu(lengths - self.m_neg, inplace=True) ** 2

        margin_loss = labels * left + self.alpha * (1. - labels) * right

        return margin_loss
