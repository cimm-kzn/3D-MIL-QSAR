import torch
import numpy as np
from torch import nn
import torch_optimizer as optim
from torch.utils.data import DataLoader
from torch.nn import Sigmoid, Sequential, Linear, ReLU
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from .mi_nets import MainNet
from .base_nets import  BaseClassifier, BaseRegressor
from .utils import  get_mini_batches, train_val_split
from typing import Sequence, Tuple, Union


class MLP(nn.Module):
    """
    Multilayer perceptron model with 3 hidden layers (2 hidden + 1 out) and ReLU nonlinearities.
    """
    def __init__(self, ndim: Sequence,  init_cuda: bool =False):
        """
        Parameters
        -----------

        ndim: Sequence
        Each entry of sequence Specifies the number of nodes in each layer and length
        of the sequence specifies number of layers
        init_cuda: bool, default is False
        Use Cuda GPU or not?
        """
        super().__init__()
        self.init_cuda = init_cuda
        self.main_net = MainNet(ndim)
        self.estimator = Linear(ndim[-1], 1)

        if self.init_cuda:
            self.main_net.cuda()
            self.estimator.cuda()


    def loss_batch(self, x_mb, y_mb, optimizer=None):
        """
         Compute loss on mini batch. NOTE: This method works only with  subclasses which initialize loss
         Parameteres
         -----------
         x_mb: torch.Tensor
         y_mb: torch.Tensor
         optimizer: Optional[torch.optim.Optimizer]
         instance of optimizer
         Returns
         -----------
         Loss per batch

         Examples
         -----------
         >>> from torch import randn, manual_seed
         >>> from torch_optimizer import Yogi
         >>> from miqsar.estimators.wrappers import MLPNetRegressor
         >>> s = manual_seed(0) # seed for reproducible net weights (assign seed to a variable to supress std output)
         >>> x_train, y_train= randn((3, 3)), randn((3,1)) # 3 molecules with 3 descriptors each (no conformers for simplicity)
         >>> net = MLPNetRegressor(ndim=(x_train[0].shape[-1], 4, 6, 4), init_cuda=False)
         >>> opt = Yogi(net.parameters())  # instantiate optimizer
         >>> loss_mb = net.loss_batch(x_train, y_train, opt)
         >>> round(loss_mb,1)
         0.1
         """
        y_out = self.forward(x_mb)
        total_loss = self.loss(y_out, y_mb)
        if optimizer is not None:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        return total_loss.item()

    def forward(self, x):
        out = self.main_net(x)
        out = self.estimator(out)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        return out

    def fit(self, x: Union[Sequence[Union[Sequence, np.array]], np.array],
            y: Union[Sequence,np.array], n_epoch: int = 100, batch_size: int = 128,
            lr: float = 0.001, weight_decay: float = 0, dropout: float = 0, verbose: bool = False) -> 'MLP':
        """
        Fit data to model.  NOTE: this method works only with  subclasses (which initialize estimator/loss)
        Parameters
        ----------
        x: array-like
        If array: array of bags of shape Nmol*Nconf*Ndescr,
        where:  Nmol - number of molecules  in dataset, Nconf - number of conformers for a given molecule,
        Ndescr - length of descriptor string  for a conformer.
        If sequence:  sequence with bags, size of a bag  (Nconf) can vary. Each entry of a bag is a descriptor
        vector for that conformer (that is not allowed to vary in length, it will be padded).

         y: array-like
        Labels for bags, array of shape Nmol (or sequence of length Nmol)

        n_epoch: int, default is 100
        Number of training epochs
        batch_size: int, default is 128
        Size of minibatch. TODO: implement check for minimal size

        lr: float, default 0.001
        Learning rate fo optimizer

        weight_decay: float, default is apply no L2 penalty (0)
        Value by which to multiply L2 penalty for optimizer

        verbose: bool, default False

        Returns
        -------
        Network with trained weights

        """
        x_train, x_val, y_train, y_val = train_val_split(x, y)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)
        if self.init_cuda:
            x_train, x_val, y_train, y_val = x_train.cuda(), x_val.cuda(), y_train.cuda(), y_val.cuda()
        optimizer = optim.Yogi(self.parameters(), lr=lr, weight_decay=weight_decay)

        val_loss = []
        for epoch in range(n_epoch):
            mb = get_mini_batches(x_train, y_train, batch_size=batch_size)
            self.train()
            for x_mb, y_mb in mb:
                loss = self.loss_batch(x_mb, y_mb, optimizer=optimizer)

            self.eval()
            with torch.no_grad():
                loss = self.loss_batch(x_val, y_val, optimizer=None)
                val_loss.append(loss)

            min_loss_idx = val_loss.index(min(val_loss))
            if min_loss_idx == epoch:
                best_parameters = self.state_dict()
                if verbose:
                    print(epoch, loss)
        self.load_state_dict(best_parameters, strict=True)
        return self

    def predict(self, x):
        x = torch.from_numpy(x.astype('float32'))
        self.eval()
        with torch.no_grad():
            if self.init_cuda:
                x = x.cuda()
            y_pred = self.forward(x)
        return np.asarray(y_pred.cpu())


class MLPNetClassifier(MLP, BaseClassifier):

    def __init__(self, ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, init_cuda=init_cuda)


class MLPNetRegressor(MLP, BaseRegressor):
    def __init__(self, ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, init_cuda=init_cuda)

class BagWrapper:
    """
    Wrapper around any network. Converts multiinstance task into single instance task by squashing dataset
     (typically, array of  Nmolecules*Nconformers*Ndescriptos) by pooling method specified, and resulting single-instance
    dataset (array of  Nmolecules*Ndescriptos) gets fed into network in training/prediction mode
    """

    def __init__(self, estimator:nn.Module, pool:str='mean') -> None:
        """
        Parameters
        -----------
        estimator:nn.Module
        any network
        pool:str='mean'
        Pooling method
        """
        self.estimator = estimator
        self.pool = pool

    def apply_pool(self, bags: Union[Sequence[Union[Sequence, np.array]], np.array]) -> np.array:
        """
        Examples
        --------
        >>> from numpy import ones
        >>> from miqsar.estimators.wrappers import BagWrapper, MLPNetRegressor
        >>> x = ones((3, 3, 3)) # 3 molecules with 3 conformers and descriptor vector of len=3
        >>> est = MLPNetRegressor(ndim=(x[0].shape[-1], 4, 6, 4), init_cuda=False)
        >>> bagwr = BagWrapper(estimator=est)
        >>> x_modified = bagwr.apply_pool(x)
        >>> x_modified # after pooling we have a single descriptor vector for each molecule
        array([[1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.]])

        Parameters
        ----------
        bags: Array-like
        Conformers of molecules represented by descriptors, i.e. 3D array-like struture
        If array: array of bags of shape Nmol*Nconf*Ndescr,
        where:  Nmol - number of molecules  in dataset, Nconf - number of conformers for a given molecule,
        Ndescr - length of descriptor string  for a conformer.
        If sequence:  sequence with bags, size of a bag  (Nconf) can vary. Each entry of a bag is a descriptor
        vector for that conformer (that is not allowed to vary in length).

        Returns
        -------
        New 2D array as a result of pooling
        """
        if self.pool == 'mean':
            bags_modified = np.asarray([np.mean(bag, axis=0) for bag in bags])
        elif self.pool == 'extreme':
            bags_max = np.asarray([np.amax(bag, axis=0) for bag in bags])
            bags_min = np.asarray([np.amin(bag, axis=0) for bag in bags])
            bags_modified = np.concatenate((bags_max, bags_min), axis=1)
        elif self.pool == 'max':
            bags_modified = np.asarray([np.amax(bag, axis=0) for bag in bags])
        elif self.pool == 'min':
            bags_modified = np.asarray([np.amin(bag, axis=0) for bag in bags])
        return bags_modified

    def fit(self, bags:Union[Sequence[Union[Sequence, np.array]], np.array], labels: Union[Sequence, np.array],
            n_epoch:int=100, batch_size:int=128,
            weight_decay:float=0, dropout:float=0,
            temp:int=1, lr:float=0.001) -> nn.Module:
        """
        NOTE: works only with estimators providing loss method
        Examples
        --------
        >>> from numpy import ones
        >>> from miqsar.estimators.wrappers import BagWrapper,MLPNetRegressor
        >>> x_train, y_train = ones((3, 3, 3)), ones(3) # toy data initialize with all 1
        >>> est = MLPNetRegressor(ndim=(x_train[0].shape[-1], 4, 6, 4), init_cuda=False)
        >>> bagwr = BagWrapper(estimator=est)
        >>> result = bagwr.fit(bags=x_train, labels=y_train, n_epoch=1, batch_size=1)

        Parameters
        ----------
        bags: Array-like
        Conformers of molecules represented by descriptors, i.e. 3D array-like struture
        If array: array of bags of shape Nmol*Nconf*Ndescr,
        where:  Nmol - number of molecules  in dataset, Nconf - number of conformers for a given molecule,
        Ndescr - length of descriptor string  for a conformer.
        If sequence:  sequence with bags, size of a bag  (Nconf) can vary. Each entry of a bag is a descriptor
        vector for that conformer (that is not allowed to vary in length).
l       Labels: array-like
        Labels for bags, array of shape Nmol (or sequence of length Nmol)
        n_epoch: int, default is 100
        Number of training epochs
        batch_size: int, default is 128
        Size of minibatch. TODO: implement check for minimal size
        lr: float, default 0.001
        Learning rate fo optimizer
        weight_decay: float, default is apply no L2 penalty (0)
        Value by which to multiply L2 penalty for optimizer
        dropout: float, default is 0
        temp: int, default 1

        Returns
        --------
        Network with trained weights
        """
        bags_modified = self.apply_pool(bags)
        self.estimator.fit(x=bags_modified, y=labels, n_epoch=n_epoch, batch_size=batch_size,
                           dropout=dropout, weight_decay=weight_decay, lr=lr)
        return self.estimator

    def predict(self, bags: Union[Sequence[Union[Sequence, np.array]], np.array]) -> np.array:
        """
        Predict unseen data by trained model.  NOTE: this method works only with  subclasses,
        because we need first to train a model

        Examples
        ----------

        """
        bags_modified = self.apply_pool(bags)
        preds = self.estimator.predict(bags_modified).flatten()
        return preds

    def name(self):
        return '{}{}'.format(self.__class__.__name__, self.pool.capitalize())


class InstanceWrapper:
    """
    Wapper around any net. Converts multiinstance task into single instance task by using each instance as a training
    input with label of its parent bag as a training output. Pooling is applied to pedictions in each bag to obtain
    single value.
    """

    def __init__(self, estimator, pool='mean'):
        self.estimator = estimator
        self.pool = pool

    def apply_pool(self, preds):
        if self.pool == 'mean':
            return np.mean(preds)
        elif self.pool == 'max':
            return np.max(preds)
        elif self.pool == 'min':
            return np.min(preds)
        else:
            print('No exist')
        return preds

    def fit(self, bags: Union[Sequence[Union[Sequence, np.array]], np.array], labels: Union[Sequence, np.array],
                n_epoch: int = 100, batch_size: int = 128,
                weight_decay: float = 0, dropout: float = 0,  lr: float = 0.001) -> "InstanceWrapper":
        """
        
        Examples
        --------
        >>> from numpy import ones
        >>> from miqsar.estimators.wrappers import InstanceWrapper, MLPNetRegressor
        >>> x_train, y_train = ones((3, 3, 3)), ones(3) # toy data initialize with all 1
        >>> est = MLPNetRegressor(ndim=(x_train[0].shape[-1], 4, 6, 4), init_cuda=False)
        >>> instwr = InstanceWrapper(estimator=est)
        >>> result = instwr.fit(x_train, y_train, n_epoch=1, batch_size=1)

        Parameters
        ----------
        bags: Array-like
        Conformers of molecules represented by descriptors, i.e. 3D array-like struture
        If array: array of bags of shape Nmol*Nconf*Ndescr,
        where:  Nmol - number of molecules  in dataset, Nconf - number of conformers for a given molecule,
        Ndescr - length of descriptor string  for a conformer.
        If sequence:  sequence with bags, size of a bag  (Nconf) can vary. Each entry of a bag is a descriptor
        vector for that conformer (that is not allowed to vary in length).
        Labels: array-like
        Labels for bags, array of shape Nmol (or sequence of length Nmol)
        n_epoch: int, default is 100
        Number of training epochs
        batch_size: int, default is 128
        Size of minibatch. TODO: implement check for minimal size
        lr: float, default 0.001
        Learning rate fo optimizer
        weight_decay: float, default is apply no L2 penalty (0)
        Value by which to multiply L2 penalty for optimizer
        dropout: float, default is 0

        Returns
        --------
        Network with trained weights
        """
        bags = np.asarray(bags)
        bags_modified = np.vstack(bags)
        labels_modified = np.hstack([float(lb) * np.array(np.ones(len(bag))) for bag, lb in zip(bags, labels)])
        self.estimator.fit(bags_modified, labels_modified, n_epoch=n_epoch, batch_size=batch_size,
                           dropout=dropout, weight_decay=weight_decay, lr=lr)
        return self.estimator

    def predict(self, bags: Union[Sequence[np.array], np.array]) -> np.array:
        """
        Predict unseen data by trained model.  NOTE: this method works only with  subclasses,
        because we need first to train a model

        Examples
        ----------
        >>> from torch import  manual_seed
        >>> from numpy import ones
        >>> from miqsar.estimators.wrappers import InstanceWrapperMLPRegressor
        >>> s = manual_seed(0) # seed for reproducible net weights (assign seed to a variable to supress std output)
        >>> x_train, y_train = ones((3, 3, 3)), ones(3) # toy data initialize with all 1
        >>> instwr = InstanceWrapperMLPRegressor(ndim=(x_train[0].shape[-1], 4, 6, 4), init_cuda=False)
        >>> _ = instwr.fit(x_train, y_train, n_epoch=2, batch_size=1)
        >>> instwr.predict(x_train)   # returns same predicitons for all datapoints (same input -> same output)
        array([-0.05271675, -0.05271675, -0.05271675], dtype=float32)
        """

        preds = [self.apply_pool(self.estimator.predict(bag.reshape(-1, bag.shape[-1]))) for bag in bags]
        return np.asarray(preds)

    def name(self):
        return '{}{}'.format(self.__class__.__name__, self.pool.capitalize())


class BagWrapperMLPRegressor(BagWrapper, BaseRegressor):

    def __init__(self, ndim:Sequence, pool:str='mean', init_cuda:bool=False):
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
        estimator = MLPNetRegressor(ndim=ndim, init_cuda=init_cuda)
        super().__init__(estimator=estimator, pool=pool)


class BagWrapperMLPClassifier(BagWrapper, BaseClassifier):

    def __init__(self, ndim: Sequence, pool: str = 'mean', init_cuda: bool = False):
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
        estimator = MLPNetClassifier(ndim=ndim, init_cuda=init_cuda)
        super().__init__(estimator=estimator, pool=pool)


class InstanceWrapperMLPRegressor(InstanceWrapper, BaseRegressor):

    def __init__(self, ndim, pool='mean', init_cuda=False):
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
        estimator = MLPNetRegressor(ndim=ndim, init_cuda=init_cuda)
        super().__init__(estimator=estimator, pool=pool)


class InstanceWrapperMLPClassifier(InstanceWrapper, BaseClassifier):

    def __init__(self, ndim, pool='mean', init_cuda=False):
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
        estimator = MLPNetClassifier(ndim=ndim, init_cuda=init_cuda)
        super().__init__(estimator=estimator, pool=pool)
