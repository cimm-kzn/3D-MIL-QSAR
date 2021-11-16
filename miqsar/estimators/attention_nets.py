import torch
from torch import nn
from torch.nn import Sequential, Linear, Sigmoid, Softmax, Tanh
from torch.nn.functional import softmax
from .base_nets import BaseRegressor, BaseClassifier, BaseNet
from .mi_nets import MainNet
from typing import Sequence, Tuple

class WeightsDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, w):
        if self.p == 0:
            return w
        d0 = [[i] for i in range(len(w))]
        d1 = w.argsort(dim=2)[:, :, :int(w.shape[2] * self.p)]
        d1 = [i.reshape(1, -1)[0].tolist() for i in d1]
        #
        w_new = w.clone()
        w_new[d0, :, d1] = 0
        #
        d1 = [i[0].nonzero().flatten().tolist() for i in w_new]
        w_new[d0, :, d1] = Softmax(dim=1)(w_new[d0, :, d1])
        return w_new


class AttentionNet(BaseNet):
    """
        This is not intended to be called directly, calling its regressor/classifier subclass will define loss and enabe
        training.
        Learns representation of bag by employing convolutional net (instance of MainNet class) and then
        aggregates for each molecule  all conformers into a single vector representing whole bag. Aggeregation is done  
        using attention weights, which are also learnt here. 

       """
    def __init__(self, ndim: Sequence, det_ndim: Sequence, init_cuda: bool = False):
        """
              Parameters
              ----------
              ndim: Sequence
              Hyperparameter for MainNet: each entry of sequence specifies the number of nodes in each layer and length
              of the sequence specifies number of layers
              det_ndim: Sequence
              Hyperparameter for attention subnet: each entry of sequence specifies the number of nodes in each layer and length
              of the sequence specifies number of layers
              init_cuda: bool, default is False
              Use Cuda GPU or not?

              """
        super().__init__(init_cuda=init_cuda)
        self.main_net = MainNet(ndim)
        self.estimator = Linear(ndim[-1], 1)
        #
        input_dim = ndim[-1]
        attention = []
        for dim in det_ndim:
            attention.append(Linear(input_dim, dim))
            attention.append(Sigmoid())
            input_dim = dim
        attention.append(Linear(input_dim, 1))
        self.detector = Sequential(*attention)

        if init_cuda:
            self.main_net.cuda()
            self.detector.cuda()
            self.estimator.cuda()

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
        Feed forward input data.
        Parameters
        ----------
        x: torch.Tensor
        m: torch.Tensor
        Returns
        --------
        Tuple with weights of conformers and tensor
        of shape Nmol*1, where Nmol is the number of molecules. The tensor is final output y, but it needs to be passed
        to sigmoid to obtain final class probabilities in case of classification (recall, this classs shouldnt be called directly,
        call regressor/classifier subclass to obtain final y).

        Examples
        --------
        >> > import torch
        >> > import numpy as np
        >> > from torch import randn
        >> > from miqsar.estimators.attention_nets import AttentionNet
        >> > x_train = randn((3, 3, 3))
        >> > at_net = AttentionNet(ndim=(x_train[0].shape[-1], 4, 6, 4), det_ndim = (4,4), init_cuda=False)
        >> > _, m = at_net.add_padding(x_train)
        >> > m = torch.from_numpy(m.astype('float32'))
        >> > _ = at_net.forward(x_train, m)  # (assign result to a variable to supress std output)

        """
        x = self.main_net(x)
        x_det = torch.transpose(m * self.detector(x), 2, 1)

        w = nn.functional.gumbel_softmax(x_det, tau=self.instance_dropout, dim=2)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = out.view(-1, 1)
        return w, out


class AttentionNetClassifier(AttentionNet, BaseClassifier):
    """
    Classifier, applying Attention based network for classification. Defines loss (by inheritance from classifier class).
    Examples
    ----------
    >>> from torch import randn, Tensor
    >>> from miqsar.estimators.attention_nets import AttentionNetClassifier
    >>> x_train, y_train = randn((3, 3, 3)), Tensor([1,0,1])
    >>> at_net = AttentionNetClassifier(ndim=(x_train[0].shape[-1], 4, 6, 4), det_ndim = (4,4), init_cuda=False)
    >>> at_net.fit(x_train, y_train, n_epoch=2, batch_size=1)
    AttentionNetClassifier(
      (main_net): Sequential(
        (0): Linear(in_features=3, out_features=4, bias=True)
        (1): ReLU()
        (2): Linear(in_features=4, out_features=6, bias=True)
        (3): ReLU()
        (4): Linear(in_features=6, out_features=4, bias=True)
        (5): ReLU()
      )
      (estimator): Linear(in_features=4, out_features=1, bias=True)
      (detector): Sequential(
        (0): Linear(in_features=4, out_features=4, bias=True)
        (1): Sigmoid()
        (2): Linear(in_features=4, out_features=4, bias=True)
        (3): Sigmoid()
        (4): Linear(in_features=4, out_features=1, bias=True)
      )
    )
      """
    def __init__(self, ndim: Sequence, det_ndim: Sequence, init_cuda: bool=False):
            """
            Parameters
            -----------

            ndim: Sequence
            Each entry of sequence Specifies the number of nodes in each layer and length
            of the sequence specifies number of layers
            det_ndim: Sequence
            init_cuda: bool, default is False
            Use Cuda GPU or not?
            """
            super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class AttentionNetRegressor(AttentionNet, BaseRegressor):
    """
    regressor, applying Attention based network for regression. Defines loss (by inheritance from regressor class).
    Examples
    ----------
    >>> from torch import randn, Tensor
    >>> from miqsar.estimators.attention_nets import AttentionNetRegressor
    >>> x_train, y_train = randn((3, 3, 3)), Tensor([1,0,1])
    >>> at_net = AttentionNetRegressor(ndim=(x_train[0].shape[-1], 4, 6, 4), det_ndim = (4,4), init_cuda=False)
    >>> at_net.fit(x_train, y_train, n_epoch=2, batch_size=1)
    AttentionNetRegressor(
      (main_net): Sequential(
        (0): Linear(in_features=3, out_features=4, bias=True)
        (1): ReLU()
        (2): Linear(in_features=4, out_features=6, bias=True)
        (3): ReLU()
        (4): Linear(in_features=6, out_features=4, bias=True)
        (5): ReLU()
      )
      (estimator): Linear(in_features=4, out_features=1, bias=True)
      (detector): Sequential(
        (0): Linear(in_features=4, out_features=4, bias=True)
        (1): Sigmoid()
        (2): Linear(in_features=4, out_features=4, bias=True)
        (3): Sigmoid()
        (4): Linear(in_features=4, out_features=1, bias=True)
      )
    )
      """
    def __init__(self, ndim: Sequence, det_ndim: Sequence, init_cuda=False):
            """
            Parameters
            -----------
            ndim: Sequence
            Each entry of sequence Specifies the number of nodes in each layer and length
            of the sequence specifies number of layers
            det_ndim: Sequence
            init_cuda: bool, default is False
            Use Cuda GPU or not?
            """
            super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)
