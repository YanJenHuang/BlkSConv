import torch.nn
from common import get_activation

"""
Convolutional Block:
conv layer + bn + activation (optional)
"""
class ConvBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 bias=False,
                 use_bn=True,
                 activation="relu"):
        super().__init__()
        self.use_bn = use_bn
        self.use_activation = (activation is not None)

        self.conv = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias)
        if self.use_bn:
            self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        if self.use_activation:
            self.activation = get_activation(activation)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x

def conv1x1_block(in_channels,
                  out_channels,
                  stride=1,
                  bias=False,
                  use_bn=True,
                  activation="relu"):
    return ConvBlock(
             in_channels=in_channels,
             out_channels=out_channels,
             kernel_size=1,
             stride=stride,
             padding=0,
             bias=bias,
             use_bn=use_bn,
             activation=activation)

def conv3x3_block(in_channels,
                  out_channels,
                  stride=1,
                  bias=False,
                  use_bn=True,
                  activation="relu"):
    return ConvBlock(
             in_channels=in_channels,
             out_channels=out_channels,
             kernel_size=3,
             stride=stride,
             padding=1,
             bias=bias,
             use_bn=use_bn,
             activation=activation)

def conv7x7_block(in_channels,
                  out_channels,
                  stride=1,
                  bias=False,
                  use_bn=True,
                  activation="relu"):
    return ConvBlock(
             in_channels=in_channels,
             out_channels=out_channels,
             kernel_size=7,
             stride=stride,
             padding=3,
             bias=bias,
             use_bn=use_bn,
             activation=activation)
