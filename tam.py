import torch
import torch.nn as nn
from math import log


class TAM(nn.Module):
    def __init__(self, n_ch, group_size, bypass=False):
        super(TAM, self).__init__()
        tam = []
        for i in range(int(log(n_ch) / log(group_size))):
            tam += [nn.Conv2d(n_ch, n_ch // group_size, 1, groups=n_ch // group_size)]
            n_ch //= group_size
            if n_ch == 1:
                break
            tam += [nn.PReLU(n_ch, init=0.0)]

        tam += [nn.Conv2d(n_ch, 1, 1)] if n_ch != 1 else []
        tam += [nn.Sigmoid()]
        self.tam = nn.Sequential(*tam)
        self.bypass = bypass

    def forward(self, x):
        if self.bypass:
            return x
        else:
            return x * self.tam(x)
