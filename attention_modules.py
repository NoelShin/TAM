import torch
import torch.nn as nn


class BAM(nn.Module):
    def __init__(self, n_ch, dilation=4, reduction_ratio=16):
        super(BAM, self).__init__()
        act = nn.ReLU(inplace=True)
        norm2d = nn.BatchNorm2d
        channel_branch = [nn.AdaptiveAvgPool2d(1),
                          View(-1),
                          nn.Linear(n_ch, n_ch // reduction_ratio),
                          act,
                          nn.Linear(n_ch // reduction_ratio, n_ch),
                          View(n_ch, 1, 1),
                          norm2d(n_ch)]

        spatial_branch = [nn.Conv2d(n_ch, n_ch // reduction_ratio, 1),
                          act]
        spatial_branch += [nn.Conv2d(n_ch // reduction_ratio, n_ch // reduction_ratio, 3, padding=4, dilation=dilation),
                           act] * 2
        spatial_branch += [nn.Conv2d(n_ch // reduction_ratio, 1, 1, bias=False),
                           norm2d(1)]

        self.channel_branch = nn.Sequential(*channel_branch)
        self.spatial_branch = nn.Sequential(*spatial_branch)

    def forward(self, x):
        return x * (1 + torch.sigmoid(self.channel_branch(x) + self.spatial_branch(x)))


class CBAM(nn.Module):
    def __init__(self, n_ch, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.CBAM = nn.Sequential(ChannelAttentionModule(n_ch, reduction_ratio),
                                  SpatialAttentionModule())

    def forward(self, x):
        return self.CBAM(x)


class ChannelAttentionModule(nn.Module):
    def __init__(self, n_ch, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        linear_A = nn.Linear(n_ch, n_ch // reduction_ratio)
        linear_B = nn.Linear(n_ch // reduction_ratio, n_ch)

        self.avg_descriptor = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                            View(-1),
                                            linear_A,
                                            nn.ReLU(inplace=True),
                                            linear_B,
                                            View(n_ch, 1, 1))

        self.max_descriptor = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                            View(-1),
                                            linear_A,
                                            nn.ReLU(inplace=True),
                                            linear_B,
                                            View(n_ch, 1, 1))

    def forward(self, x):
        return x * torch.sigmoid(self.avg_descriptor(x) + self.max_descriptor(x))


class ChannelAxisPool(nn.Module):
    def __init__(self, pool='avg'):
        super(ChannelAxisPool, self).__init__()
        assert pool in ['avg', 'max', 'var'], print("Invalid type {}. Choose among ['avg', 'max', 'var']".format(pool))
        self.pool = pool

    def forward(self, x):
        if self.pool == 'avg':
            return torch.mean(x, dim=1, keepdim=True)
        elif self.pool == 'max':
            return torch.max(x, dim=1, keepdim=True)[0]
        else:
            return torch.var(x, dim=1, keepdim=True)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.avg_descriptor = ChannelAxisPool('avg')
        self.max_descriptor = ChannelAxisPool('max')
        self.block = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False),
                                   nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
                                   nn.Sigmoid())

    def forward(self, x):
        return x * self.block(torch.cat((self.avg_descriptor(x), self.max_descriptor(x)), dim=1))


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, n_ch, reduction_ratio=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.block = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                   View(-1),
                                   nn.Linear(n_ch, n_ch // reduction_ratio, bias=False),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(n_ch // reduction_ratio, n_ch, bias=False),
                                   View(n_ch, 1, 1),
                                   nn.Sigmoid())

    def forward(self, x):
        return x * self.block(x)


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)
