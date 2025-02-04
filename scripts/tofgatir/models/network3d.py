from turtle import forward
import torch
from easydict import EasyDict
from copy import deepcopy


class ConvBlock3d(torch.nn.Sequential):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = EasyDict(kwargs)
        self.add_module('conv', torch.nn.Conv3d(
            self.kwargs.in_channels,
            self.kwargs.out_channels,
            self.kwargs.kernel_size,
            padding=self.kwargs.padding
        ))
        self.add_module('norm', self._create_norm())
        self.add_module('relu', torch.nn.ReLU())
        self.add_module('dp', torch.nn.Dropout3d(self.kwargs.dropout))

    def _create_norm(self):
        Norm = getattr(torch.nn, self.kwargs.norm.type)
        norm_kwargs = deepcopy(self.kwargs.norm.kwargs)
        if Norm == torch.nn.GroupNorm:
            norm_kwargs.num_channels = self.kwargs.out_channels
        elif Norm == torch.nn.BatchNorm3d:
            norm_kwargs.num_features = self.kwargs.out_channels
        return Norm(**norm_kwargs)


class Network3d(torch.nn.Sequential):
    def __init__(self, **kwargs):
        kwargs = EasyDict(kwargs)
        super().__init__()
        self.cb0 = ConvBlock3d(
            in_channels=kwargs.in_channels,
            out_channels=kwargs.channels,
            kernel_size=3,
            padding=0,
            norm=kwargs.norm,
            dropout=kwargs.dropout
        )
        for i in range(kwargs.num_blocks - 1):
            conv = ConvBlock3d(
                in_channels=kwargs.channels,
                out_channels=kwargs.channels,
                kernel_size=3,
                padding=0,
                norm=kwargs.norm,
                dropout=kwargs.dropout
            )
            self.add_module(f'cb{i+1}', conv)
        self.out = torch.nn.Conv3d(kwargs.channels, kwargs.out_channels, 1)

    def forward(self, x, **kwargs):
        result = {'pred': super().forward(x)}
        return result
