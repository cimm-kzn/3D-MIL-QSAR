import torch
from torch import nn
from torch.nn import Sequential, Linear, Sigmoid, Softmax, Tanh
from torch.nn.functional import softmax
from .base_nets import BaseRegressor, BaseClassifier, BaseNet
from .mi_nets import MainNet


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
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
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

    def forward(self, x, m):
        x = self.main_net(x)
        x_det = torch.transpose(m * self.detector(x), 2, 1)

        w = Softmax(dim=2)(x_det)
        w = WeightsDropout(p=self.dropout)(w)

        x = torch.bmm(w, x)
        out = self.estimator(x)
        if isinstance(self, BaseClassifier):
            out = Sigmoid()(out)
        out = out.view(-1, 1)
        return w, out


class AttentionNetClassifier(AttentionNet, BaseClassifier):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)


class AttentionNetRegressor(AttentionNet, BaseRegressor):
    def __init__(self, ndim=None, det_ndim=None, init_cuda=False):
        super().__init__(ndim=ndim, det_ndim=det_ndim, init_cuda=init_cuda)
