import torch
from easydict import EasyDict
from copy import deepcopy
from ..utils import map_data


class ConvBlock(torch.nn.Sequential):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = EasyDict(kwargs)
        conv_cls = getattr(torch.nn, self.kwargs.conv)
        self.add_module('conv', conv_cls(
            self.kwargs.in_channels,
            self.kwargs.out_channels,
            self.kwargs.kernel_size,
            padding=self.kwargs.padding
        ))
        self.add_module('norm', self._create_norm())
        self.add_module('relu', torch.nn.ReLU())

    def _create_norm(self):
        Norm = getattr(torch.nn, self.kwargs.norm.type)
        norm_kwargs = deepcopy(self.kwargs.norm.kwargs)
        if Norm is torch.nn.GroupNorm:
            norm_kwargs.num_channels = self.kwargs.out_channels
        elif Norm is torch.nn.BatchNorm2d or Norm is torch.nn.BatchNorm3d:
            norm_kwargs.num_features = self.kwargs.out_channels
        return Norm(**norm_kwargs)


class ContractingBlock(torch.nn.Sequential):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = EasyDict(kwargs)
        conv0 = ConvBlock(
            in_channels=self.kwargs.in_channels,
            out_channels=self.kwargs.mid_channels,
            kernel_size=self.kwargs.kernel_size,
            padding=self.kwargs.padding,
            norm=self.kwargs.norm,
            conv=self.kwargs.conv
        )
        self.add_module('conv0', conv0)
        conv1 = ConvBlock(
            in_channels=self.kwargs.mid_channels,
            out_channels=self.kwargs.out_channels,
            kernel_size=self.kwargs.kernel_size,
            padding=self.kwargs.padding,
            norm=self.kwargs.norm,
            conv=self.kwargs.conv
        )
        self.add_module('conv1', conv1)
        dp_cls = getattr(torch.nn, self.kwargs.dropout.type)
        self.add_module('dp', dp_cls(**self.kwargs.dropout.kwargs))


class ExpandingBlock(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = EasyDict(kwargs)

        conv0 = ConvBlock(
            in_channels=self.kwargs.in_channels + self.kwargs.shortcut_channels,
            out_channels=self.kwargs.out_channels,
            kernel_size=self.kwargs.kernel_size,
            padding=self.kwargs.padding,
            norm=self.kwargs.norm,
            conv=self.kwargs.conv
        )
        self.add_module('conv0', conv0)
        conv1 = ConvBlock(
            in_channels=self.kwargs.out_channels,
            out_channels=self.kwargs.out_channels,
            kernel_size=self.kwargs.kernel_size,
            padding=self.kwargs.padding,
            norm=self.kwargs.norm,
            conv=self.kwargs.conv
        )
        self.add_module('conv1', conv1)
        dp_cls = getattr(torch.nn, self.kwargs.dropout.type)
        self.add_module('dp', dp_cls(**self.kwargs.dropout.kwargs))

    def forward(self, x, shortcut):
        output = torch.cat((x, shortcut), dim=1) # concat channels
        output = self.conv0(output)
        output = self.conv1(output)
        output = self.dp(output)
        return output
    
class OutBlock(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = EasyDict(kwargs)

        conv_cls = getattr(torch.nn, self.kwargs.conv)
        self.add_module('conv', conv_cls(self.kwargs.in_channels, self.kwargs.out_channels, 1))
        self.add_module('relu', torch.nn.ReLU())
        # self.add_module('tanh', torch.nn.Tanh())

    def forward(self, x):
        output = self.conv(x)
        output = self.relu(output)
        # output = self.tanh(output)
        return output

class UNet(torch.nn.Module):
    """UNet with split output branches.

    Attributes:
        in_channels (int): The number of the channels of the input.
        out_channels (int): The number of the channels of the output.
        num_trans_down (int): The number of transition down. This number
            controls the "depth" of the network.
        first_channels (int): The number of output channels of the input block.
            This number controls the "width" of the networ.
        max_channels (int): The maximum number of tensor channels.
        output_levels (int or list[int]): The indices of the levels that
            give outputs. The top level is 0.

    """
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = EasyDict(kwargs)

        # encoding/contracting
        self.cb0 = self._create_ib(
            self.kwargs.in_channels, self.kwargs.first_channels,
            self.kwargs.first_channels
            # (self.kwargs.in_channels + self.kwargs.first_channels) // 2,
        )   # first,out,mid
        out_channels = self.kwargs.first_channels
        in_channels = out_channels
        for i in range(self.kwargs.num_trans_down):
            out_channels = self._calc_out_channels(in_channels)
            self.add_module(f'td{i}', self._create_td())
            cb = self._create_cb(in_channels, out_channels, out_channels)
            self.add_module(f'cb{i + 1}', cb)
            in_channels = out_channels

        # decoding/expanding
        for i in reversed(range(self.kwargs.num_trans_down)):
            out_channels = getattr(self, f'cb{i}').kwargs.out_channels
            if self._is_split_eb(i):
                tu = self._create_tu(in_channels, out_channels)
                self.add_module(f'tu{i}', tu)
                eb = self._create_eb(out_channels, out_channels, out_channels)
                self.add_module(f'eb{i}', eb)
            else:
                tus = torch.nn.ModuleList()
                ebs = torch.nn.ModuleList()
                for _ in range(self.kwargs.branches):
                    tus.append(self._create_tu(in_channels, out_channels))
                    ebs.append(self._create_eb(
                        out_channels, out_channels, out_channels))
                self.add_module(f'tu{i}', tus)
                self.add_module(f'eb{i}', ebs)
            in_channels = out_channels

        outs = torch.nn.ModuleList()
        for _ in range(self.kwargs.branches):
            outs.append(self._create_out(in_channels))
        self.add_module('out', outs)    # negative?

    def _calc_out_channels(self, in_channels):
        return min(in_channels * 2, self.kwargs.max_channels)

    def _is_split_eb(self, i):
        return i >= self.kwargs.split_ebs

    def _create_ib(self, in_channels, out_channels, mid_channels):
        return self._create_cb(in_channels, out_channels, mid_channels)

    def _create_cb(self, in_channels, out_channels, mid_channels):
        return ContractingBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            mid_channels=mid_channels,
            kernel_size=self.kwargs.kernel_size,
            padding=self.kwargs.padding,
            norm=self.kwargs.norm,
            dropout=self.kwargs.dropout,
            conv=self.kwargs.conv,
        )

    def _create_td(self):
        return getattr(torch.nn, self.kwargs.td)(2)

    def _create_tu(self, in_channels, out_channels):
        return torch.nn.Sequential(
            getattr(torch.nn, self.kwargs.conv)(in_channels, out_channels, 1),
            torch.nn.Upsample(scale_factor=2)
        )

    def _create_eb(self, in_channels, shortcut_channels, out_channels):
        return ExpandingBlock(
            in_channels=in_channels,
            shortcut_channels=shortcut_channels,
            out_channels=out_channels,
            kernel_size=self.kwargs.kernel_size,
            padding=self.kwargs.padding,
            norm=self.kwargs.norm,
            dropout=self.kwargs.dropout,
            conv=self.kwargs.conv,
        )

    def _create_out(self, in_channels):
        # conv_cls = getattr(torch.nn, self.kwargs.conv)
        # return conv_cls(in_channels, self.kwargs.out_channels, 1)
        return OutBlock(in_channels=in_channels,
                    out_channels=self.kwargs.out_channels,
                    kernel_size=1,
                    conv=self.kwargs.conv)

    def forward(self, x, **kwargs):
        # encoding/contracting
        output = x
        shortcuts = list()
        for i in range(self.kwargs.num_trans_down):
            output = getattr(self, f'cb{i}')(output)
            shortcuts.append(output)
            output = getattr(self, f'td{i}')(output)

        # bridge
        i = self.kwargs.num_trans_down
        output = getattr(self, f'cb{i}')(output)

        # decoding/expanding
        for i in reversed(range(self.kwargs.num_trans_down)):
            if self._is_split_eb(i):        # 1,2 
                output = getattr(self, f'tu{i}')(output)
                output = getattr(self, f'eb{i}')(output, shortcuts[i])
            else:   # 3,4
                if not isinstance(output, list):
                    output = [output] * self.kwargs.branches
                for j in range(self.kwargs.branches):
                    output[j] = getattr(self, f'tu{i}')[j](output[j])
                    output[j] = getattr(self, f'eb{i}')[j](output[j], shortcuts[i])

        # output
        for j in range(self.kwargs.branches):
            output[j] = getattr(self, 'out')[j](output[j])  #32，1？，60，60 original
            # out_ac=torch.nn.ReLU()  # added
            # output[j] = out_ac(output[j])
            # output[j]=(output[j]-torch.min(output[j]))/(torch.max(output[j])-torch.min(output[j]))
        output = self._postproc(output, **kwargs)

        return output

    def _postproc(self, output, **kwargs):
        output = torch.cat(output, dim=1)   #dim1:2x3->2x6
        return {'pred': output}


class T1Eq(torch.nn.Module):
    def __init__(self, use_abs=False, scale=1000):
        super().__init__()
        self.use_abs = use_abs
        self.scale = scale

    def forward(self, x, params):
        ndim = x[0].ndim - 2
        ti = params[(..., 0) + (None,) * ndim]
        tr = params[(..., 1) + (None,) * ndim]
        x = [self.scale * torch.clip(xx, min=1e-5) for xx in x]
        term1 = torch.exp(-ti / x[1])
        term2 = torch.exp(-tr / x[1])
        out = x[0] * (1 - 2 * term1 + term2)
        if self.use_abs:
            out = torch.abs(out)
        return out


class T1Eq2(torch.nn.Module):
    def __init__(self, use_abs=False, scale=1000):
        super().__init__()
        self.use_abs = use_abs
        self.scale = scale

    def forward(self, x, params):
        ndim = x[0].ndim - 2
        ti = params[(..., 0) + (None,) * ndim]
        tr = params[(..., 1) + (None,) * ndim]
        x = [self.scale * xx for xx in x]
        term1 = torch.exp(-ti / x[1])
        term2 = torch.exp(-tr / x[1])
        out = x[0] * (1 - 2 * term1 + term2)
        if self.use_abs:
            out = torch.abs(out)
        return out


class UNetEq(UNet):
    def __init__(self, **kwargs):
        eq_kwargs = kwargs.pop('eq')
        super().__init__(**kwargs)
        self.kwargs.eq = eq_kwargs
        self.eq = T1Eq(**eq_kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.out:
            torch.nn.init.constant_(layer.bias, 1.0)

    def _postproc(self, output, **kwargs):
        return {
            'interm': torch.cat(output, dim=1) * self.eq.scale,
            'pred': self.eq(output, **kwargs),
        }


class UNetEq2(UNet):
    def __init__(self, **kwargs):
        eq_kwargs = kwargs.pop('eq')
        super().__init__(**kwargs)
        self.kwargs.eq = eq_kwargs
        self.eq = T1Eq2(**eq_kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.out:
            torch.nn.init.constant_(layer.bias, 1.0)

    def _postproc(self, output, **kwargs):
        return {
            'interm': torch.cat(output, dim=1) * self.eq.scale,
            'pred': self.eq(output, **kwargs),
        }
    

class ConvBlocktanh(torch.nn.Sequential):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = EasyDict(kwargs)
        conv_cls = getattr(torch.nn, self.kwargs.conv)
        self.add_module('conv', conv_cls(
            self.kwargs.in_channels,
            self.kwargs.out_channels,
            self.kwargs.kernel_size,
            padding=self.kwargs.padding
        ))
        self.add_module('norm', self._create_norm())
        self.add_module('lrelu', torch.nn.LeakyReLU())

    def _create_norm(self):
        Norm = getattr(torch.nn, self.kwargs.norm.type)
        norm_kwargs = deepcopy(self.kwargs.norm.kwargs)
        if Norm is torch.nn.GroupNorm:
            norm_kwargs.num_channels = self.kwargs.out_channels
        elif Norm is torch.nn.BatchNorm2d or Norm is torch.nn.BatchNorm3d:
            norm_kwargs.num_features = self.kwargs.out_channels
        return Norm(**norm_kwargs)


class ContractingBlocktanh(torch.nn.Sequential):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = EasyDict(kwargs)
        conv0 = ConvBlocktanh(
            in_channels=self.kwargs.in_channels,
            out_channels=self.kwargs.mid_channels,
            kernel_size=self.kwargs.kernel_size,
            padding=self.kwargs.padding,
            norm=self.kwargs.norm,
            conv=self.kwargs.conv
        )
        self.add_module('conv0', conv0)
        conv1 = ConvBlocktanh(
            in_channels=self.kwargs.mid_channels,
            out_channels=self.kwargs.out_channels,
            kernel_size=self.kwargs.kernel_size,
            padding=self.kwargs.padding,
            norm=self.kwargs.norm,
            conv=self.kwargs.conv
        )
        self.add_module('conv1', conv1)
        dp_cls = getattr(torch.nn, self.kwargs.dropout.type)
        self.add_module('dp', dp_cls(**self.kwargs.dropout.kwargs))


class ExpandingBlocktanh(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = EasyDict(kwargs)

        conv0 = ConvBlocktanh(
            in_channels=self.kwargs.in_channels + self.kwargs.shortcut_channels,
            out_channels=self.kwargs.out_channels,
            kernel_size=self.kwargs.kernel_size,
            padding=self.kwargs.padding,
            norm=self.kwargs.norm,
            conv=self.kwargs.conv
        )
        self.add_module('conv0', conv0)
        conv1 = ConvBlocktanh(
            in_channels=self.kwargs.out_channels,
            out_channels=self.kwargs.out_channels,
            kernel_size=self.kwargs.kernel_size,
            padding=self.kwargs.padding,
            norm=self.kwargs.norm,
            conv=self.kwargs.conv
        )
        self.add_module('conv1', conv1)
        dp_cls = getattr(torch.nn, self.kwargs.dropout.type)
        self.add_module('dp', dp_cls(**self.kwargs.dropout.kwargs))

    def forward(self, x, shortcut):
        output = torch.cat((x, shortcut), dim=1) # concat channels
        output = self.conv0(output)
        output = self.conv1(output)
        output = self.dp(output)
        return output

class OutBlocktanh(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = EasyDict(kwargs)

        conv_cls = getattr(torch.nn, self.kwargs.conv)
        self.add_module('conv', conv_cls(self.kwargs.in_channels, self.kwargs.out_channels, 1))
        self.add_module('tanh', torch.nn.Tanh())

    def forward(self, x):
        output = self.conv(x)
        output = self.tanh(output)
        return output

class UNettanh(torch.nn.Module):
    """UNet with split output branches.

    Attributes:
        in_channels (int): The number of the channels of the input.
        out_channels (int): The number of the channels of the output.
        num_trans_down (int): The number of transition down. This number
            controls the "depth" of the network.
        first_channels (int): The number of output channels of the input block.
            This number controls the "width" of the networ.
        max_channels (int): The maximum number of tensor channels.
        output_levels (int or list[int]): The indices of the levels that
            give outputs. The top level is 0.

    """
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = EasyDict(kwargs)

        # encoding/contracting
        self.cb0 = self._create_ib(
            self.kwargs.in_channels, self.kwargs.first_channels,
            self.kwargs.first_channels
            # (self.kwargs.in_channels + self.kwargs.first_channels) // 2,
        )   # first,out,mid
        out_channels = self.kwargs.first_channels
        in_channels = out_channels
        for i in range(self.kwargs.num_trans_down):
            out_channels = self._calc_out_channels(in_channels)
            self.add_module(f'td{i}', self._create_td())
            cb = self._create_cb(in_channels, out_channels, out_channels)
            self.add_module(f'cb{i + 1}', cb)
            in_channels = out_channels

        # decoding/expanding
        for i in reversed(range(self.kwargs.num_trans_down)):
            out_channels = getattr(self, f'cb{i}').kwargs.out_channels
            if self._is_split_eb(i):
                tu = self._create_tu(in_channels, out_channels)
                self.add_module(f'tu{i}', tu)
                eb = self._create_eb(out_channels, out_channels, out_channels)
                self.add_module(f'eb{i}', eb)
            else:
                tus = torch.nn.ModuleList()
                ebs = torch.nn.ModuleList()
                for _ in range(self.kwargs.branches):
                    tus.append(self._create_tu(in_channels, out_channels))
                    ebs.append(self._create_eb(
                        out_channels, out_channels, out_channels))
                self.add_module(f'tu{i}', tus)
                self.add_module(f'eb{i}', ebs)
            in_channels = out_channels

        outs = torch.nn.ModuleList()
        for _ in range(self.kwargs.branches):
            outs.append(self._create_out(in_channels))
        self.add_module('out', outs)    # negative?
        # self.add_module('tanh', torch.nn.Tanh())    #added


    def _calc_out_channels(self, in_channels):
        return min(in_channels * 2, self.kwargs.max_channels)

    def _is_split_eb(self, i):
        return i >= self.kwargs.split_ebs

    def _create_ib(self, in_channels, out_channels, mid_channels):
        return self._create_cb(in_channels, out_channels, mid_channels)

    def _create_cb(self, in_channels, out_channels, mid_channels):
        return ContractingBlocktanh(
            in_channels=in_channels,
            out_channels=out_channels,
            mid_channels=mid_channels,
            kernel_size=self.kwargs.kernel_size,
            padding=self.kwargs.padding,
            norm=self.kwargs.norm,
            dropout=self.kwargs.dropout,
            conv=self.kwargs.conv,
        )

    def _create_td(self):
        return getattr(torch.nn, self.kwargs.td)(2)

    def _create_tu(self, in_channels, out_channels):
        return torch.nn.Sequential(
            getattr(torch.nn, self.kwargs.conv)(in_channels, out_channels, 1),
            torch.nn.Upsample(scale_factor=2)
        )

    def _create_eb(self, in_channels, shortcut_channels, out_channels):
        return ExpandingBlocktanh(
            in_channels=in_channels,
            shortcut_channels=shortcut_channels,
            out_channels=out_channels,
            kernel_size=self.kwargs.kernel_size,
            padding=self.kwargs.padding,
            norm=self.kwargs.norm,
            dropout=self.kwargs.dropout,
            conv=self.kwargs.conv,
        )

    def _create_out(self, in_channels):
        # conv_cls = getattr(torch.nn, self.kwargs.conv)
        # return conv_cls(in_channels, self.kwargs.out_channels, 1) # original
        return OutBlocktanh(in_channels=in_channels,
                            out_channels=self.kwargs.out_channels,
                            kernel_size=1,
                            conv=self.kwargs.conv)
    
    def forward(self, x, **kwargs):
        # encoding/contracting
        output = x
        vmax=output.max()
        vmin=output.min()
        shortcuts = list()
        for i in range(self.kwargs.num_trans_down):
            output = getattr(self, f'cb{i}')(output)
            # output1= getattr(self, f'cb{i}.conv0')(output)
            # output2= getattr(self, f'cb{i}.conv1')(output1)
            # output= getattr(self, f'cb{i}.dp')(output2)
            vmax=output.max()
            vmin=output.min()
            shortcuts.append(output)
            output = getattr(self, f'td{i}')(output)
            vmax=output.max()
            vmin=output.min()

        # bridge
        i = self.kwargs.num_trans_down
        output = getattr(self, f'cb{i}')(output)
        vmax=output.max()
        vmin=output.min()

        # decoding/expanding
        for i in reversed(range(self.kwargs.num_trans_down)):
            if self._is_split_eb(i):        # 1,2 
                output = getattr(self, f'tu{i}')(output)
                output = getattr(self, f'eb{i}')(output, shortcuts[i])
                vmax=output.max()
                vmin=output.min()
            else:   # 3,4
                if not isinstance(output, list):
                    output = [output] * self.kwargs.branches
                for j in range(self.kwargs.branches):
                    output[j] = getattr(self, f'tu{i}')[j](output[j])
                    output[j] = getattr(self, f'eb{i}')[j](output[j], shortcuts[i])
                    vmax=output[j].max()
                    vmin=output[j].min()

        # output
        for j in range(self.kwargs.branches):
            output[j] = getattr(self, 'out')[j](output[j])  #32，1？，60，60
            # out_ac=torch.nn.Tanh()  # added
            # output[j] = out_ac(output[j])
            # output[j]=(output[j]-torch.min(output[j]))/(torch.max(output[j])-torch.min(output[j]))
        output = self._postproc(output, **kwargs)
        vmax=output['pred'].max()
        vmin=output['pred'].min()

        return output

    def _postproc(self, output, **kwargs):
        output = torch.cat(output, dim=1)   #dim1:2x3->2x6
        return {'pred': output}