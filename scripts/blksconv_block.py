import torch.nn
from blksconv import BlkSConv2d
from common import get_activation

def blksconv(in_channels, out_channels, kernel_size, num_basis, blk_depth, stride=1, padding=0, bias=False):
    return BlkSConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            num_basis=num_basis,
            blk_depth=blk_depth,
            padding=padding,
            bias=bias)

def blksconv3x3(in_channels, out_channels, num_basis, blk_depth, stride=1, padding=1, bias=False):
    return BlkSConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            num_basis=num_basis,
            blk_depth=blk_depth,
            padding=padding,
            bias=bias)

"""
BlkSConv Block:
blksconv layer + bn + activation (optional)
"""
class BlkSConvBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 num_basis,
                 blk_depth,
                 bias=False,
                 use_bn=True,
                 activation="relu"):
        super().__init__()
        self.use_bn = use_bn
        self.use_activation = (activation is not None)
        
        self.subspace_conv = BlkSConv2d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                num_basis=num_basis,
                                blk_depth=blk_depth,
                                padding=padding,
                                bias=bias)

        if self.use_bn:
            self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        if self.use_activation:
            self.activation = get_activation(activation)
        
    def forward(self, x):
        x = self.subspace_conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x

def blksconv3x3_block(in_channels,
                          out_channels,
                          stride=1,
                          num_basis=1,
                          blk_depth=1,
                          bias=False,
                          use_bn=True,
                          activation="relu"):
    return BlkSConv2d(
             in_channels=in_channels,
             out_channels=out_channels,
             kernel_size=3,
             stride=stride,
             padding=1,
             num_basis=num_basis,
             blk_depth=blk_depth,
             bias=bias,
             use_bn=use_bn,
             activation=activation)

def blksconv7x7_block(in_channels,
                          out_channels,
                          stride=1,
                          num_basis=1,
                          blk_depth=1,
                          bias=False,
                          use_bn=True,
                          activation="relu"):
    return BlkSConv2d(
             in_channels=in_channels,
             out_channels=out_channels,
             kernel_size=7,
             stride=stride,
             padding=3,
             num_basis=num_basis,
             blk_depth=blk_depth,
             bias=bias,
             use_bn=use_bn,
             activation=activation)