import torch
import numpy as np
from torch import nn
import torch_optimizer as optim
from torch.utils.data import DataLoader
from torch.nn import Sigmoid
from sklearn.model_selection import train_test_split
from .utils import MBSplitter

class BaseClassifier:
    def loss(self, y_pred, y_true):
        total_loss = nn.BCELoss(reduction='mean')(y_pred, y_true.reshape(-1, 1))
        return total_loss


class BaseRegressor:
    def loss(self, y_pred, y_true):
        total_loss = nn.MSELoss(reduction='mean')(y_pred, y_true.reshape(-1, 1))
        return total_loss


class BaseNet(nn.Module):
    def __init__(self, net=None, init_cuda=False):
        super().__init__()
        self.net = net
        self.init_cuda = init_cuda

        if self.net and self.init_cuda:
            self.net.cuda()


    def name(self):
        return self.__class__.__name__

    def add_padding(self, x):
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

    def train_val_split(self, x, y, val_size=0.2, random_state=42):
        x, y = np.asarray(x), np.asarray(y)
        x, m = self.add_padding(x)

        x_train, x_val, y_train, y_val, m_train, m_val = train_test_split(x, y, m, test_size=val_size,
                                                                          random_state=random_state)
        x_train, y_train, m_train = self.array_to_tensor(x_train, y_train, m_train)
        x_val, y_val, m_val = self.array_to_tensor(x_val, y_val, m_val)

        return x_train, x_val, y_train, y_val, m_train, m_val

    def get_mini_batches(self, x, y, m, batch_size=16):
        data = MBSplitter(x, y, m)
        mb = DataLoader(data, batch_size=batch_size, shuffle=True)
        return mb

    def array_to_tensor(self, x, y, m):

        x = torch.from_numpy(x.astype('float32'))
        y = torch.from_numpy(y.astype('float32'))
        m = torch.from_numpy(m.astype('float32'))
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if self.init_cuda:
            x, y, m = x.cuda(), y.cuda(), m.cuda()
        return x, y, m

    def loss_batch(self, x_mb, y_mb, m_mb, optimizer=None):
        w_out, y_out = self.forward(x_mb, m_mb)
        total_loss = self.loss(y_out, y_mb)
        if optimizer is not None:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        return total_loss.item()

    def forward(self, x, m):
        x = m * self.main_net(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(x)
        out = out.view(-1, 1)
        return None, out

    def fit(self, x, y, n_epoch=100, batch_size=128, lr=0.001, weight_decay=0, instance_dropout=0.95, verbose=False):
        self.instance_dropout = instance_dropout

        x_train, x_val, y_train, y_val, m_train, m_val = self.train_val_split(x, y)
        optimizer = optim.Yogi(self.parameters(), lr=lr, weight_decay=weight_decay)

        val_loss = []
        for epoch in range(n_epoch):
            mb = self.get_mini_batches(x_train, y_train, m_train, batch_size=batch_size)
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

    def predict(self, x):
        x, m = self.add_padding(np.asarray(x))
        x = torch.from_numpy(x.astype('float32'))
        m = torch.from_numpy(m.astype('float32'))
        self.eval()
        with torch.no_grad():
            if self.init_cuda:
                x, m = x.cuda(), m.cuda()
            w, y_pred = self.forward(x, m)
        return np.asarray(y_pred.cpu())

    def get_instance_weights(self, x):
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


