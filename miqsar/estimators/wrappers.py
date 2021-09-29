import torch
import numpy as np
from torch import nn
import torch_optimizer as optim
from torch.utils.data import DataLoader
from torch.nn import Sigmoid, Sequential, Linear, ReLU
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from .mi_nets import MainNet


class MBSplitter(Dataset):
    def __init__(self, x, y):
        super(MBSplitter, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.y)


class BaseClassifier:
    def loss(self, y_pred, y_true):
        total_loss = nn.BCELoss(reduction='mean')(y_pred, y_true.reshape(-1, 1))
        return total_loss


class BaseRegressor:
    def loss(self, y_pred, y_true):
        total_loss = nn.MSELoss(reduction='mean')(y_pred, y_true.reshape(-1, 1))
        return total_loss


class MLP(nn.Module):
    def __init__(self, ndim=None, init_cuda=False):
        super().__init__()
        self.init_cuda = init_cuda
        self.main_net = MainNet(ndim)
        self.estimator = Linear(ndim[-1], 1)

        if self.init_cuda:
            self.main_net.cuda()
            self.estimator.cuda()

    def train_val_split(self, x, y, val_size=0.2, random_state=42):
        x, y = np.asarray(x), np.asarray(y)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_size, random_state=random_state)
        x_train, y_train = self.array_to_tensor(x_train, y_train)
        x_val, y_val = self.array_to_tensor(x_val, y_val)

        return x_train, x_val, y_train, y_val

    def get_mini_batches(self, x, y, batch_size=16):
        data = MBSplitter(x, y)
        mb = DataLoader(data, batch_size=batch_size, shuffle=True)
        return mb

    def array_to_tensor(self, x, y):

        x = torch.from_numpy(x.astype('float32'))
        y = torch.from_numpy(y.astype('float32'))
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if self.init_cuda:
            x, y = x.cuda(), y.cuda()
        return x, y

    def loss_batch(self, x_mb, y_mb, optimizer=None):
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

    def fit(self, x, y, n_epoch=100, batch_size=128, lr=0.001, weight_decay=0, dropout=0, verbose=False):

        x_train, x_val, y_train, y_val = self.train_val_split(x, y)
        optimizer = optim.Yogi(self.parameters(), lr=lr, weight_decay=weight_decay)

        val_loss = []
        for epoch in range(n_epoch):
            mb = self.get_mini_batches(x_train, y_train, batch_size=batch_size)
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

    def __init__(self, estimator, pool='mean'):
        self.estimator = estimator
        self.pool = pool

    def apply_pool(self, bags):
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

    def fit(self, bags, labels, n_epoch=100, batch_size=128, weight_decay=0, dropout=0, temp=1, lr=0.001):
        bags_modified = self.apply_pool(bags)
        self.estimator.fit(bags_modified, labels, n_epoch=n_epoch, batch_size=batch_size,
                           dropout=dropout, weight_decay=weight_decay, lr=lr)
        return self.estimator

    def predict(self, bags):
        bags_modified = self.apply_pool(bags)
        preds = self.estimator.predict(bags_modified).flatten()
        return preds

    def name(self):
        return '{}{}'.format(self.__class__.__name__, self.pool.capitalize())


class InstanceWrapper:

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

    def fit(self, bags, labels, n_epoch=100, batch_size=128, dropout=0, weight_decay=0, lr=0.001):
        bags = np.asarray(bags)
        bags_modified = np.vstack(bags)
        labels_modified = np.hstack([float(lb) * np.array(np.ones(len(bag))) for bag, lb in zip(bags, labels)])
        self.estimator.fit(bags_modified, labels_modified, n_epoch=n_epoch, batch_size=batch_size,
                           dropout=dropout, weight_decay=weight_decay, lr=lr)
        return self.estimator

    def predict(self, bags):
        preds = [self.apply_pool(self.estimator.predict(bag.reshape(-1, bag.shape[-1]))) for bag in bags]
        return np.asarray(preds)

    def name(self):
        return '{}{}'.format(self.__class__.__name__, self.pool.capitalize())


class BagWrapperMLPRegressor(BagWrapper, BaseRegressor):

    def __init__(self, ndim=None, pool='mean', init_cuda=False):
        estimator = MLPNetRegressor(ndim=ndim, init_cuda=init_cuda)
        super().__init__(estimator=estimator, pool=pool)


class BagWrapperMLPClassifier(BagWrapper, BaseClassifier):

    def __init__(self, ndim=None, pool='mean', init_cuda=False):
        estimator = MLPNetClassifier(ndim=ndim, init_cuda=init_cuda)
        super().__init__(estimator=estimator, pool=pool)


class InstanceWrapperMLPRegressor(InstanceWrapper, BaseRegressor):

    def __init__(self, ndim=None, pool='mean', init_cuda=False):

        estimator = MLPNetRegressor(ndim=ndim, init_cuda=init_cuda)
        super().__init__(estimator=estimator, pool=pool)


class InstanceWrapperMLPClassifier(InstanceWrapper, BaseClassifier):

    def __init__(self, ndim=None, pool='mean', init_cuda=False):
        estimator = MLPNetClassifier(ndim=ndim, init_cuda=init_cuda)
        super().__init__(estimator=estimator, pool=pool)
