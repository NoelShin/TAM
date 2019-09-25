import torch
import torch.nn as nn


class ChannelAxisPool(nn.Module):
    def __init__(self, pool='channelwise_conv', n_ch=None):
        super(ChannelAxisPool, self).__init__()
        assert pool in ['avg', 'channelwise_conv', 'max', 'var'], print("Invalid type {}. Choose among ['avg', 'max', 'var']".format(pool))
        if pool == 'channelwise_conv':
            assert n_ch, print("To use channelwise_conv, you need to clarify n_ch argument.")
            self.conv = nn.Conv2d(n_ch, 1, 1)
        self.pool = pool

    def forward(self, x):
        if self.pool == 'avg':
            return torch.mean(x, dim=1, keepdim=True)

        elif self.pool == 'channelwise_conv':
            return self.conv(x)

        elif self.pool == 'max':
            return torch.max(x, dim=1, keepdim=True)[0]

        else:
            return torch.var(x, dim=1, keepdim=True)


class MixedSeparableConv2d(nn.Module):
    def __init__(self, n_ch, stride=1, dilation=1, bias=True):
        super(MixedSeparableConv2d, self).__init__()
        self.n_ch = n_ch
        self.conv3 = nn.Conv2d(n_ch // 4, n_ch // 4, 3, stride=stride, dilation=dilation,
                               padding=(3 + 2 * (dilation - 1)) // 2, groups=n_ch // 4, bias=bias)
        self.conv5 = nn.Conv2d(n_ch // 4, n_ch // 4, 5, stride=stride, dilation=dilation,
                               padding=(5 + 4 * (dilation - 1)) // 2, groups=n_ch // 4, bias=bias)
        self.conv7 = nn.Conv2d(n_ch // 4, n_ch // 4, 7, stride=stride, dilation=dilation,
                               padding=(7 + 6 * (dilation - 1)) // 2, groups=n_ch // 4, bias=bias)
        self.conv9 = nn.Conv2d(n_ch // 4, n_ch // 4, 9, stride=stride, dilation=dilation,
                               padding=(9 + 8 * (dilation - 1)) // 2, groups=n_ch // 4, bias=bias)

    def forward(self, x):
        return torch.cat((self.conv3(x[:, :self.n_ch // 4, ...]),
                          self.conv5(x[:, self.n_ch // 4:self.n_ch // 2, ...]),
                          self.conv7(x[:, self.n_ch // 2:self.n_ch * 3 // 4, ...]),
                          self.conv9(x[:, self.n_ch * 3 // 4:, ...])),
                          dim=1)


class TAM(nn.Module):
    def __init__(self, n_ch, conversion_factor=4):
        super(TAM, self).__init__()
        self.tam = nn.Sequential()
        for i in range(conversion_factor):
            self.tam.add_module("1x1Conv{}".format(i), nn.Conv2d(n_ch, n_ch // 2, 1, groups=n_ch // 2))
            if i != conversion_factor - 1:
                self.tam.add_module("Act0{}".format(i), nn.PReLU(n_ch // 2, init=0))
            n_ch //= 2

        self.tam.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, x):
        return x * self.ccm(x)


class DBAM(nn.Module):
    def __init__(self, n_ch, kernel_size=(), padding=()):
        super(DBAM, self).__init__()
        dbam = []
        for i in range(len(kernel_size)):
            n_ch //= 4
            dbam += [nn.PixelShuffle(2),
                     nn.Conv2d(n_ch, n_ch, kernel_size[i], dilation=2 ** (i + 1), padding=padding[i], groups=n_ch),
                     nn.ReLU(True)]

        dbam += [nn.Conv2d(n_ch, 1, 1)]
        self.dbam = nn.Sequential(*dbam)

    def forward(self, x):
        print(x.shape, self.dbam(x).shape)
        print()
        return x * torch.sigmoid(self.dbam(x))


class AddActConv(nn.Module):
    def __init__(self, n_ch, kernel_size, padding, dilation):
        super(AddActConv, self).__init__()
        self.act_conv = nn.Sequential(#nn.BatchNorm2d(n_ch),
                                      nn.ReLU(True),
                                      nn.Conv2d(n_ch, n_ch, kernel_size, padding=padding, dilation=dilation, groups=n_ch, bias=False),
                                      #nn.BatchNorm2d(n_ch),
                                      nn.ReLU(True),
                                      nn.Conv2d(n_ch, n_ch, kernel_size, padding=padding, dilation=dilation, groups=n_ch, bias=False))

    def forward(self, x):
        return x + self.act_conv(x)


class ChannelDuplication(nn.Module):
    def __init__(self, factor):
        super(ChannelDuplication, self).__init__()
        self.factor = factor

    def forward(self, x):
        return torch.cat(tuple(x for _ in range(self.factor)), dim=1)


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


class Crop(nn.Module):
    def __init__(self, boundary=(0, 0)):
        super(Crop, self).__init__()
        self.crop = True if boundary != (0, 0) else False
        self.even = True if (boundary[0] % 2) == 0 else False
        self.y = boundary[0]
        self.x = boundary[1]

    def forward(self, x):
        if self.crop:
            if self.even:

                return x[:, :, self.y // 2:-(self.y // 2), self.x // 2:-(self.x // 2)]
            else:
                return x[:, :, :-self.y, :-self.x]
        else:
            return x


class Replicate(nn.Module):
    def __init__(self, conversion_factor):
        super(Replicate, self).__init__()
        self.conversion_factor = conversion_factor

    def forward(self, x):
        return torch.repeat_interleave(x, 2 ** self.conversion_factor, dim=1)


class Print(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x
