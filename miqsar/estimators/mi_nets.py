import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU, Softmax, Sigmoid
from .base_nets import BaseNet, BaseRegressor, BaseClassifier
from typing import Sequence, Tuple

class MainNet:
    """
    Abstract class not intended to be invoked directly.
    Main net for multiple-instance learning via convolution on conformers. (conformation ensembles are hereafter termed "bags".)
    Linear transform maps each conformer to new hidden space of dimensionality hd (hd1,hd2,hd3..) This is perfomed seqentially ndim-1 times,
    with RelU nonlinearity after each linear layer (linear layer can be viewed as a set
    of 1-dimensional convolution filters). The result is learnt representation of bags of shape Nmols*Nconf*Nhlast, where
    Nmols - number of molecules, Nconf -number of conformers and Nhlast - dimensionality of last hidden layer.

    This learnt representation (as opposed to original one) is assumed to help better predict property studied.
    THe same net can be used for single instance learning, in which case it essentially is MLP.
    """

    def __new__(cls, ndim: Sequence):
        """
        Parameters
        -----------
        ndim: Sequence
        Each entry of sequence Specifies the number of nodes in each layer and length
        of the sequence specifies number of layers
        Returns
        --------
        MainNet instance
        """
        ind, hd1, hd2, hd3 = ndim
        net = Sequential(Linear(ind, hd1),
                         ReLU(),
                         Linear(hd1, hd2),
                         ReLU(),
                         Linear(hd2, hd3),
                         ReLU())

        return net



class Pooling(nn.Module):
    """
    Pooling layer aggregates a bag (in its current representation)
    into a single vector: whole-molecule representation, using  mean/max/lse operator
    """
    def __init__(self, pool: str = 'mean') -> None:
        """
        Parameters
        -----------
         pool: str
         Name of opertaor to use for pooling
        """
        super().__init__()
        self.pool = pool

    def forward(self, x: torch.Tensor, m: torch.Tensor) ->  torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
        Input data
        m: torch.Tensor
        Mask tensor, indicating with ones exisiting conformers, and with zeros - values coming from padding
        Returns
        --------
        torch.Tensor
        Result of pooling: single vector representation per bag (vector can have only one value, if input
        dimensionality was R (1)). The shape of output: Nmol*Ndescr, where Nmol - number of molecules (bags),
        Ndescr - number of descriptors, i.e. dimensionalty of current input embeddidng

        Examples
        --------
        >>> import torch
        >>> from torch import Tensor, ones
        >>> from miqsar.estimators.mi_nets import Pooling
        >>> # create tensor with 4 molecules, 3 conformers per molecule and 3 descriptors. Also we init m with ones,
        ... # (for compatibitlity only, indicates which conformers are real and which come from zero-padding)
        ... x,m = torch.Tensor([[[3, 3, 3],[1, 1, 1],[2, 2, 2]],[[3, 3, 3],[1, 1, 1],[2, 2, 2]], [[3, 3, 3],[1, 1, 1],[2, 2, 2]],
        ... [[3, 3, 3],[1, 1, 1], [2, 2, 2]]]), torch.ones([4,3,1])
        >>> p = Pooling(pool="mean")
        >>> p.forward(x,m)
        tensor([[2., 2., 2.],
                [2., 2., 2.],
                [2., 2., 2.],
                [2., 2., 2.]])
        """
        x = m * x 
        if self.pool == 'mean':
            out = x.sum(axis=1) / m.sum(axis=1)
        elif self.pool == 'max':
            out = x.max(dim=1)[0]
        elif self.pool == 'lse':
            out = x.exp().sum(dim=1).log()
        return out

    def extra_repr(self):
        return 'Pooling(out_dim=1)'


class BagNet(BaseNet):
    """
     This is not intended to be called directly, calling its regressor/classifier subclass will define loss and enabe
     training.
     Learns representation  by convolutional net (instance of MainNet class) and then
     aggregates for each molecule  all conformers into a single vector (by method defined in pool parameter), representing whole bag.

    """
    def __init__(self, ndim:Sequence, pool:str='mean', init_cuda:bool=False) -> None:
        """
        Parameters
        ----------
        ndim: Sequence
        Hyperparameter for MainNet: each entry of sequence specifies the number of nodes in each layer and length
        of the sequence specifies number of layers
        pool: str, default is None
        Pooling method to use
        init_cuda: bool, default is False
        Use Cuda GPU or not?

        """
        super().__init__(init_cuda=init_cuda)
        self.main_net = MainNet(ndim)
        self.pooling = Pooling(pool)
        self.estimator = Linear(ndim[-1], 1)

        if self.init_cuda:
            self.main_net.cuda()
            self.estimator.cuda()

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> Tuple[None, torch.Tensor]:
        """
        Feed forward input data.
        Parameters
        ----------
        x: torch.Tensor
        m: torch.Tensor
        Returns
        --------
        Tuple with None (standing for weights of conformers, which are not applicable here), and tensor
        of shape Nmol*1, where Nmol is the number of molecules. The tensor is final output y, but it needs to be passed
        to sigmoid to obtain final class probabilities in case of classification (this classs shouldnt be called directly,
        call regressor/classifier subclass to obtain final y).


        Examples
        --------
        >>> import torch
        >>> import numpy as np
        >>> from torch import randn
        >>> from miqsar.estimators.mi_nets import BagNet
        >>> x_train = randn((3, 3, 3))
        >>> bag_net = BagNet(ndim=(x_train[0].shape[-1], 4, 6, 4), init_cuda=False)
        >>> _, m = bag_net.add_padding(x_train)
        >>> m = torch.from_numpy(m.astype('float32'))
        >>> _ = bag_net.forward(x_train, m) # (assign result to a variable to supress std output)

        """
        out = self.main_net(x)
        out = self.pooling(out, m)
        out = self.estimator(out)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        return None, out


class InstanceNet(BaseNet):
    """
     This is not intended to be called directly, calling its regressor/classifier subclass will define which
     estimator to apply before final aggregation and loss to enable trainig.
     Learns representation  by convolutional net (instance of MainNet class) and then applies conformer-level
     regressor/classifier, and finally aggegates conformer-level values into single value per molecule.

    """

    def __init__(self, ndim:Sequence, pool:str='mean', init_cuda:bool=False) -> None:
        """
        Parameters
        ----------
        ndim: sequence
        Each entry of sequence specifies the number of nodes in each layer and length
        of the sequence specifies number of layers
        pool: str, default is None
        Pooling method to use
        init_cuda: bool, default is False
        Use Cuda GPU or not?
        """

        super().__init__(init_cuda=init_cuda)
        self.main_net = Sequential(MainNet(ndim), Linear(ndim[-1], 1))
        self.pooling = Pooling(pool)

        if self.init_cuda:
            self.main_net.cuda()

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Feed forward input data.
        Parameters
        ----------
        x: torch.Tensor
        m: torch.Tensor
        Returns
        --------
        Tuple with  weights of conformers, and tensor
        of shape Nmol*1, where Nmol is the number of molecules. The tensor is final output y, but it needs to be passed
        to sigmoid to obtain final class probabilities in case of classification (this classs shouldnt be called directly,
        call regressor/classifier subclass to obtained final y).

        Examples
        ----------
        >>> import torch
        >>> import numpy as np
        >>> from torch import randn
        >>> from miqsar.estimators.mi_nets import InstanceNet
        >>> x_train = randn((3, 3, 3))
        >>> instance_net = InstanceNet(ndim=(x_train[0].shape[-1], 4, 6, 4), init_cuda=False)
        >>> _, m = instance_net.add_padding(x_train)
        >>> m = torch.from_numpy(m.astype('float32'))
        >>> _ = instance_net.forward(x_train, m) # (assign result to a variable to supress std output)

        """
        out = self.main_net(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        w = Softmax(dim=1)(m * out)
        w = w.view(w.shape[0], w.shape[-1], w.shape[1])
        out = self.pooling(out, m)
        return w, out


class BagNetClassifier(BagNet, BaseClassifier):
    """
    Classifier, applying BagNet method for classification. Defines loss (by inheritance from classifier class).
    Examples
    ----------
    >>> from torch import randn, Tensor
    >>> from miqsar.estimators.mi_nets import BagNetClassifier
    >>> x_train, y_train = randn((3, 3, 3)), Tensor([1,0,1])
    >>> bag_net = BagNetClassifier(ndim=(x_train[0].shape[-1], 4, 6, 4), init_cuda=False)
    >>> bag_net.fit(x_train, y_train, n_epoch=2, batch_size=1)
    BagNetClassifier(
      (main_net): Sequential(
        (0): Linear(in_features=3, out_features=4, bias=True)
        (1): ReLU()
        (2): Linear(in_features=4, out_features=6, bias=True)
        (3): ReLU()
        (4): Linear(in_features=6, out_features=4, bias=True)
        (5): ReLU()
      )
      (pooling): Pooling(Pooling(out_dim=1))
      (estimator): Linear(in_features=4, out_features=1, bias=True)
    )
    """
    def __init__(self, ndim: Sequence, pool:str='mean', init_cuda:bool=False) -> None:
        """
        Parameters
        -----------

        ndim: Sequence
        Each entry of sequence Specifies the number of nodes in each layer and length
        of the sequence specifies number of layers
        pool: str, default is None
        Pooling method to use
        init_cuda: bool, default is False
        Use Cuda GPU or not?
        """
        super().__init__(ndim=ndim, pool=pool, init_cuda=init_cuda)


class BagNetRegressor(BagNet, BaseRegressor):
    """
    Regressor, applying BagNet method for regression. Defines loss (by inheritance from regressor class).

    Examples
    ----------
    >>> from torch import randn
    >>> from miqsar.estimators.mi_nets import BagNetRegressor
    >>> x_train, y_train = randn((3, 3, 3)), randn(3)
    >>> bag_net = BagNetRegressor(ndim=(x_train[0].shape[-1], 4, 6, 4), init_cuda=False)
    >>> bag_net.fit(x_train, y_train, n_epoch=2, batch_size=1)
    BagNetRegressor(
      (main_net): Sequential(
        (0): Linear(in_features=3, out_features=4, bias=True)
        (1): ReLU()
        (2): Linear(in_features=4, out_features=6, bias=True)
        (3): ReLU()
        (4): Linear(in_features=6, out_features=4, bias=True)
        (5): ReLU()
      )
      (pooling): Pooling(Pooling(out_dim=1))
      (estimator): Linear(in_features=4, out_features=1, bias=True)
    )
    """
    def __init__(self, ndim:Sequence, pool:str='mean', init_cuda:bool=False) -> None:
        """
        Parameters
        -----------
        ndim: Sequence
        Each entry of sequence Specifies the number of nodes in each layer and length
        of the sequence specifies number of layers
        pool: str, default is None
        Pooling method to use
        init_cuda: bool, default is False
        Use Cuda GPU or not?
        """
        super().__init__(ndim=ndim, pool=pool, init_cuda=init_cuda)


class InstanceNetClassifier(InstanceNet, BaseClassifier):
    """
    Classifier, applying InstanceNet method for classification. Defines loss.

    Examples
    ----------
    >>> from torch import randn, Tensor
    >>> from miqsar.estimators.mi_nets import InstanceNetClassifier
    >>> x_train, y_train = randn((3, 3, 3)), Tensor([1,0,1])
    >>> inst_net = InstanceNetClassifier(ndim=(x_train[0].shape[-1], 4, 6, 4), init_cuda=False)
    >>> inst_net.fit(x_train, y_train, n_epoch=2, batch_size=1)
    InstanceNetClassifier(
      (main_net): Sequential(
        (0): Sequential(
          (0): Linear(in_features=3, out_features=4, bias=True)
          (1): ReLU()
          (2): Linear(in_features=4, out_features=6, bias=True)
          (3): ReLU()
          (4): Linear(in_features=6, out_features=4, bias=True)
          (5): ReLU()
        )
        (1): Linear(in_features=4, out_features=1, bias=True)
      )
      (pooling): Pooling(Pooling(out_dim=1))
    )
    """
    def __init__(self, ndim:Sequence, pool:str='mean', init_cuda:bool=False) -> None:
        """
        Parameters
        -----------
        ndim: Sequence
        Each entry of sequence Specifies the number of nodes in each layer and length
        of the sequence specifies number of layers
        pool: str, default is None
        Pooling method to use
        init_cuda: bool, default is False
        Use Cuda GPU or not?
        """
        super().__init__(ndim=ndim, pool=pool, init_cuda=init_cuda)


class InstanceNetRegressor(InstanceNet, BaseRegressor):
    """
    Regressor, applying InstanceNet method for regression. Defines loss.
    Examples
    ----------
    >>> from torch import randn, Tensor
    >>> from miqsar.estimators.mi_nets import InstanceNetRegressor
    >>> x_train, y_train = randn((3, 3, 3)), randn(3)
    >>> inst_net = InstanceNetRegressor(ndim=(x_train[0].shape[-1], 4, 6, 4), init_cuda=False)
    >>> inst_net.fit(x_train, y_train, n_epoch=2, batch_size=1)
    InstanceNetRegressor(
      (main_net): Sequential(
        (0): Sequential(
          (0): Linear(in_features=3, out_features=4, bias=True)
          (1): ReLU()
          (2): Linear(in_features=4, out_features=6, bias=True)
          (3): ReLU()
          (4): Linear(in_features=6, out_features=4, bias=True)
          (5): ReLU()
        )
        (1): Linear(in_features=4, out_features=1, bias=True)
      )
      (pooling): Pooling(Pooling(out_dim=1))
    )
    """
    def __init__(self, ndim:Sequence, pool:str='mean', init_cuda:bool=False) -> None:
        """
        Parameters
        -----------
        ndim: Sequence
        Each entry of sequence Specifies the number of nodes in each layer and length
        of the sequence specifies number of layers
        pool: str, default is None
        Pooling method to use
        init_cuda: bool, default is False
        Use Cuda GPU or not?
        """
        super().__init__(ndim=ndim, pool=pool, init_cuda=init_cuda)
