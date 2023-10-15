import torch
import torch.nn as nn
import numpy as np
from common import get_activation

class BlkSConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 num_basis,
                 blk_depth,
                 activation=None,
                 padding=0,
                 bias=False):
        super().__init__()
        self.use_activation = (activation is not None)
        if self.use_activation:
            self.activation = get_activation(activation)
        self.padding = padding
        self.bias = bias

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_basis = num_basis
        self.blk_depth = blk_depth
        self.num_blocks = self.in_channels // self.blk_depth

        self.group_pointwise = nn.Conv2d(in_channels=self.num_blocks,
                                         out_channels=self.out_channels*self.num_basis,
                                         kernel_size=1,
                                         bias=self.bias)
        self.group_share_blocks_depthwise = nn.Conv2d(in_channels=self.blk_depth*(self.out_channels*self.num_basis),
                                                      out_channels=self.out_channels*self.num_basis,
                                                      kernel_size=self.kernel_size,
                                                      stride=self.stride,
                                                      groups=(self.out_channels*self.num_basis),
                                                      padding=self.padding,
                                                      bias=self.bias)
        self.group_basis_summation_layer = self.group_basis_summation

    def _tensor2flatten_tensor(self, x):
        # B, C, H, W
        self.B, self.C, self.H, self.W = x.shape
        x = x.view(self.B, self.blk_depth, self.C//self.blk_depth, self.H, self.W)
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(self.B, self.C//self.blk_depth, self.H*self.blk_depth, self.W)
        return x
    
    def _flatten_tensor2tensor(self, x):
        return x.view(self.B,
                   self.blk_depth*(self.out_channels*self.num_basis),
                   self.H,
                   self.W)
    
    def group_basis_summation(self, x):
        _,_,H,W = x.shape
        return torch.sum(input=x.view(self.B, 
                                      self.num_basis, self.out_channels,
                                      H, W),
                         dim=1,
                         keepdim=False)

    def forward(self, x):
        x = self._tensor2flatten_tensor(x)
        x = self.group_pointwise(x)

        if self.use_activation:
            x = self.activation(x)

        x = self._flatten_tensor2tensor(x)
        x = self.group_share_blocks_depthwise(x)
        x = self.group_basis_summation_layer(x)

        return x
    
    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, num_basis={self.num_basis}, blk_depth={self.blk_depth}, stride={self.stride}, padding={self.padding}'