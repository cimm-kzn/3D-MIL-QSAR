import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, as_tensor, Tensor
from torch.utils.data import TensorDataset, DataLoader
from typing import Sequence, Tuple, Union, List
from sklearn.model_selection import train_test_split

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_mini_batches(*tensors: Tensor, batch_size: int = 16) -> DataLoader:
        """
        Batch generator

        Examples
        -----------
        >>> from torch import randn, manual_seed
        >>> from miqsar.estimators.utils import get_mini_batches
        >>> s = manual_seed(0) # seed for reproducible net weights (assign seed to a variable to supress std output)
        >>> x_train, y_train= randn((3, 3, 3)), randn(3)
        >>> mb = get_mini_batches( x_train, y_train, batch_size=1)
        >>> list(mb)
        [[tensor([[[-0.4927,  0.0537,  0.6181],
                 [-0.4128, -0.8411, -2.3160],
                 [-0.1023,  0.7924, -0.2897]]]), tensor([-1.1088])], [tensor([[[-1.1258, -1.1524, -0.2506],
                 [-0.4339,  0.8487,  0.6920],
                 [-0.3160, -2.1152,  0.3223]]]), tensor([-0.2251])], [tensor([[[-1.2633,  0.3500,  0.2660],
                 [ 0.1665,  0.8744, -0.1435],
                 [-0.1116, -0.6136,  0.0316]]]), tensor([2.3466])]]

        Parameters
        -----------
        *tensors
        Sequence of:
        Descriptors tensor; Labels (target values) tensor; if needed: Mask tensor m (shape of m: Nmol*max(Nconf)*1)
        batch_size: int

        Returns
        -----------
        Loader of minibatches
         """
        data = TensorDataset(*tensors)
        mb = DataLoader(data, batch_size=batch_size, shuffle=True)
        return mb

def train_val_split(*seqs: Union[Sequence, np.array],
                        val_size: float = 0.2, random_state: int =42) -> List[Tensor]:
        """
        Wrapper around sklearn train_val_split for compatibility with torch.Tensors

        Examples
        --------
        >>> from numpy.random import standard_normal, seed
        >>> from miqsar.estimators.utils import train_val_split
        >>> s = seed(0) # seed for reproducible net weights (assign seed to a variable to supress std output)
        >>> x, y = standard_normal((3, 3, 3)), standard_normal(3)
        >>> train_val_split(x, y)
        [tensor([[[ 0.4106,  0.1440,  1.4543],
                 [ 0.7610,  0.1217,  0.4439],
                 [ 0.3337,  1.4941, -0.2052]],
        <BLANKLINE>
                [[ 0.3131, -0.8541, -2.5530],
                 [ 0.6536,  0.8644, -0.7422],
                 [ 2.2698, -1.4544,  0.0458]]]), tensor([[[ 1.7641,  0.4002,  0.9787],
                 [ 2.2409,  1.8676, -0.9773],
                 [ 0.9501, -0.1514, -0.1032]]]), tensor([1.5328, 1.4694]), tensor([-0.1872])]

        Parameters
        -----------
        *seqs: Union[Sequence, np.array]
        Sequence of sequences/np.arrays containing:
        bags (=sets of conformers, one set for each molecule)
        y:  labels (one for each molecule)
        m: mask array, optional
        val_size: float, default is .2
        Validation set portion
        random_state: int

        Returns
        -----------
        List of  torch.tensors: x,y, and padding mask m (if was supplied) -
        for both train and val splits
        """
      
        splitted_arrays = train_test_split(*seqs, test_size=val_size, random_state=random_state)
        splitted_tensors = [as_tensor(i, dtype=torch.float32) for i in splitted_arrays]# as_tensor makes no copy, if dtype and device same

        return splitted_tensors

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
