import torch
import numpy as np
from torch import nn, Tensor
import torch_optimizer as optim
from torch.utils.data import DataLoader
from torch.nn import Sigmoid
from sklearn.model_selection import train_test_split
from utils import get_mini_batches, train_val_split
from typing import Sequence, Tuple, Optional, List, Union

class BaseClassifier:
    """
    Abstract base class for classifiers.
    This is intended only for convenience of subclass implementations
    and should not be invoked directly.
    """
    def loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Parameters
        ----------
        y_pred: Tensor
        predicted  values (activity)
        y_true: Tensor
        true target values (activity)
        Returns
        ----------
        Tensor
        Loss, currently, binary cross entropy
        """
        total_loss = nn.BCELoss(reduction='mean')(y_pred, y_true.reshape(-1, 1))
        return total_loss


class BaseRegressor:
    """
    Abstract base class for regressors.
    This is intended only for convenience of subclass implementations
    and should not be invoked directly.
    """
    def loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Computes loss (currently, it's only mean-squared error)
        Parameters
        ----------
        y_pred: Tensor
        predicted  values (activity)
        y_true: Tensor
        true target values (activity)

        Returns
        ----------
        Tensor
        Loss
        """ 
        total_loss = nn.MSELoss(reduction='mean')(y_pred, y_true.reshape(-1, 1))
        return total_loss


class BaseNet(nn.Module):
    """
    Abstract base class for all MIL models.
    This is intended only for convenience of subclass implementations
    and should not be invoked directly.
    """
    def __init__(self, net: Optional[nn.Module] = None, init_cuda: bool =False):
        """
        Parameters
        ----------
        net: nn.Module
        PyTorch model object.
        init_cuda: bool
        Should cuda GPUs be used or not?

        """
        super().__init__()
        self.net = net
        self.init_cuda = init_cuda
      
        if self.net and self.init_cuda:
            self.net.cuda()


    def name(self):
        return self.__class__.__name__

    def add_padding(self, x: Union[Sequence, np.array]) -> Tuple[np.array,np.array]:
        """
        Adds zero-padding to each  bag in x (sequence of bags) to bring x to tensor of shape Nmol*max(Nconf)*Ndescr,
        where: Nconf - number of conformers for a given molecule,
        Ndescr - length of descriptor string, Nmol - number of molecules  in dataset
        Examples
        --------
        >>> import numpy as np
        >>> from miqsar.estimators.base_nets import BaseNet
        >>> net = BaseNet()
        >>> x_train = [[[1, 1],[1,1]],[[1, 1]]] # 2 molecules, one with 2 conformers and the other with only 1 conformer
        >>> _, m = net.add_padding(x_train)
        >>> m
        array([[[1.],
                [1.]],
        <BLANKLINE>
               [[1.],
                [0.]]])

        Parameters
         -----------
         x:  Union[Sequence, np.array]
         Sequence of bags (sets of conformers, one set for each molecule)
         Returns
         -----------
         Tuple of 2 tensors: new padded  tensor x and   mask tensor m (shape of m: Nmol*max(Nconf)*1): each row populated with
          either 1 where conformer exists, or 0 where conformer didnt exist and zeros were added.
            """
        bag_size = max(len(i) for i in x)
        mask = np.ones((len(x), bag_size, 1))

        out = []
        for i, bag in enumerate(x):
            bag = np.asarray(bag)
            if len(bag) < bag_size:
                mask[i][len(bag):] = 0
                padding = np.zeros((bag_size - bag.shape[0], bag.shape[1]))
                bag = np.vstack((bag, padding))
            out.append(bag)
        out_bags = np.asarray(out)
        return out_bags, mask

    def loss_batch(self, x_mb: Tensor, y_mb: Tensor, m_mb: Tensor, optimizer: Optional[torch.optim.Optimizer]=None) -> float:
        """
        Compute loss on mini batch. NOTE: This method works only with  subclasses which initialize main_net and loss
        Parameteres
        -----------
        x_mb: torch.Tensor
        y_mb: torch.Tensor
        m_mb: torch.Tensor
        optimizer: Optional[torch.optim.Optimizer]
        instance of optimizer
        Returns
        -----------
        Loss per batch

        Examples
        -----------
        >>> from torch import randn, manual_seed, from_numpy
        >>> from torch_optimizer import Yogi
        >>> from miqsar.estimators.mi_nets import BagNetRegressor
        >>> s = manual_seed(0) # seed for reproducible net weights (assign seed to a variable to supress std output)
        >>> x_train, y_train= randn((3, 3, 3)), randn(3)
        >>> bag_net = BagNetRegressor(ndim=(x_train[0].shape[-1], 4, 6, 4), init_cuda=False)
        >>> _, m = bag_net.add_padding(x_train)
        >>> m = from_numpy(m).float()
        >>> opt = Yogi(bag_net.parameters())  # instantiate optimizer
        >>> loss_mb = bag_net.loss_batch(x_train, y_train,m, opt)
        >>> round(loss_mb,2)
        2.8
        """
        w_out, y_out = self.forward(x_mb, m_mb)
        total_loss = self.loss(y_out, y_mb)
        if optimizer is not None:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        return total_loss.item()

    def forward(self, x, m):
        """
         NOTE: this method works only with  subclasses, which initialize main_net
        """
        x = m * self.main_net(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(x)
        out = out.view(-1, 1)
        return None, out

    def fit(self, x: Union[Sequence[Union[Sequence, np.array]], np.array], y: Union[Sequence, np.array],
            n_epoch: int = 100, batch_size: int = 128, lr: float = 0.001,
            weight_decay: float = 0, instance_dropout: float = 0.95, verbose: bool = False)-> 'BaseNet':
        """
        Main fit method. fit data to model.  NOTE: his method works only with  subclasses
        Parameters
        ----------
        x: array-like
        If array: array of bags of shape Nmol*Nconf*Ndescr,
        where:  Nmol - number of molecules  in dataset, Nconf - number of conformers for a given molecule,
        Ndescr - length of descriptor string  for a conformer.
        If sequence:  sequence with bags, size of a bag  (Nconf) can vary (if varies, will be padded).
         Each entry of a bag is a descriptor
        vector for that conformer (that is not allowed to vary in length).
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
        instance_dropout: float, default is 0.95
        Randomly zeroes some of the instances with probability 1-instance_dropout (during training)
        using samples from a Bernoulli distribution.
        verbose: bool, default False

        Returns
        --------
        Network with trained weights        

        """

        self.instance_dropout = instance_dropout
        x, m = self.add_padding(x)
        x_train, x_val, y_train, y_val, m_train, m_val = train_val_split(x, y, m) 
        if y.ndim == 1: # convert 1d array into 2d ("column-vector")
            y_train = y_train.reshape(-1, 1)
        if y_val.ndim == 1:  # convert 1d array into 2d ("column-vector")
            y_val = y_val.reshape(-1, 1)
        if self.init_cuda:
            x_train, x_val, y_train, y_val, m_train, m_val  = x_train.cuda(), x_val.cuda(), y_train.cuda(), y_val.cuda(), \
                                                              m_train.cuda(), m_val.cuda()
        optimizer = optim.Yogi(self.parameters(), lr=lr, weight_decay=weight_decay)

        val_loss = []
        for epoch in range(n_epoch):
            mb = get_mini_batches(x_train, y_train, m_train, batch_size=batch_size)
            self.train()
            for x_mb, y_mb, m_mb in mb:
                loss = self.loss_batch(x_mb, y_mb, m_mb, optimizer=optimizer)

            self.eval()
            with torch.no_grad():
                loss = self.loss_batch(x_val, y_val, m_val, optimizer=None)
                val_loss.append(loss)

            min_loss_idx = val_loss.index(min(val_loss))
            if min_loss_idx == epoch:
                best_parameters = self.state_dict()
                if verbose:
                    print(epoch, loss)
        self.load_state_dict(best_parameters, strict=True)
        return self

    def predict(self, x: Union[Sequence[Union[Sequence, np.array]], np.array]) -> np.array:
        """
        Main predict method. Predict unseen data by trained model.  NOTE: his method works only with  subclasses,
        because we need first to train a model
        Parameters
        ----------
        x: array-like
        Conformers of molecules represented by descriptors, i.e. 3D array-like struture
        If array: array of bags of shape Nmol*Nconf*Ndescr,
        where:  Nmol - number of molecules  in dataset, Nconf - number of conformers for a given molecule,
        Ndescr - length of descriptor string  for a conformer.
        If sequence:  sequence with bags, size of a bag  (Nconf) can vary (if varies, will be padded). Each entry of a bag is a descriptor
        vector for that conformer (that is not allowed to vary in length).

        Returns
        --------
        Network's predicted values, array of shape Nmol

        Examples
        ----------
        >>> from torch import ones, manual_seed
        >>> from miqsar.estimators.mi_nets import BagNetRegressor
        >>> s = manual_seed(0) # seed for reproducible net weights (assign seed to a variable to supress std output)
        >>> x_train, y_train = ones((3, 3, 3)), ones(3) # toy data initialize with all 1
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
        >>> bag_net.predict(x_train)   # returns same predicitons for all datapoints (same input -> same output)
        array([[-0.07465197],
               [-0.07465197],
               [-0.07465197]], dtype=float32)

        """
        x, m = self.add_padding(np.asarray(x))
        x = torch.from_numpy(x.astype('float32'))
        m = torch.from_numpy(m.astype('float32'))
        self.eval()
        with torch.no_grad():
            if self.init_cuda:
                x, m = x.cuda(), m.cuda()
            w, y_pred = self.forward(x, m)
        return np.asarray(y_pred.cpu())

    def get_instance_weights(self, x: Union[Sequence[Union[Sequence, np.array]], np.array]) -> List[np.array]:
        """
        NOTE: his method works only with  multiinstance architectures whose last layer is linear
        Parameters
        ----------
        x: array-like
        If array: array of bags of shape Nmol*Nconf*Ndescr,
        where:  Nmol - number of molecules  in dataset, Nconf - number of conformers for a given molecule,
        Ndescr - length of descriptor string  for a conformer.
        If sequence:  sequence with bags, size of a bag  (Nconf) can vary. Each entry of a bag is a descriptor
        vector for that conformer (that is not allowed to vary in length, it will be padded).

        Returns
        --------
        Weights from the last linear layer (if present in given network architecture)
        Returned list contains weights of conformers for each bag (len(list) = number of molecules, len(sublist/array) =
         number of conformers)

        """
        x, m = self.add_padding(np.asarray(x))
        x = torch.from_numpy(x.astype('float32'))
        m = torch.from_numpy(m.astype('float32'))
        self.eval()
        with torch.no_grad():
            if self.init_cuda:
                x, m = x.cuda(), m.cuda()
            w, y_pred = self.forward(x, m)
        w = w.view(w.shape[0], w.shape[-1]).cpu()
        w = [np.asarray(i[j.bool().flatten()]) for i, j in zip(w, m)]
        return w


