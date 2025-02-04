import torch
import numpy as np
from collections import defaultdict
from torch.nn.modules import MSELoss, L1Loss, CrossEntropyLoss, SmoothL1Loss
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from pytorch_msssim  import  ssim ,  ms_ssim ,  SSIM ,  MS_SSIM 
from torchvision.transforms import Resize
import torch.nn.functional as F
import torchvision

from piq import SSIMLoss, MultiScaleSSIMLoss


from .utils import cal_std_mse,cal_std_ce

class PSNR(torch.nn.Module):
    def forward(self, pred, truth, mask=None):
        # truth = truth.abs()
        # pred = pred.abs()
        psnrs = list()
        for i in range(truth.shape[1]):
            sub_truth = truth[:, i : i + 1, ...]
            sub_pred = pred[:, i : i + 1, ...]
            max_val = torch.max(sub_truth) + 1e-5
            if mask is not None:
                sub_mask = mask.expand(sub_pred.shape)
                sub_pred = sub_pred[sub_mask]
                sub_truth = sub_truth[sub_mask]
            mse = torch.nn.functional.mse_loss(sub_pred, sub_truth) + 1e-5
            psnrs.append(10 * torch.log10(max_val * max_val / mse).item())
        return np.mean(psnrs)

class PSNR_mre(torch.nn.Module):
    def forward(self, pred, truth, mask=None):
        # truth = truth.abs()
        # pred = pred.abs()
        psnrs = list()
        for i in range(truth.shape[1]):
            sub_truth = truth[:, i : i + 1, ...]
            sub_pred = pred[:, i : i + 1, ...]
            max_val = torch.max(sub_truth) + 1e-5
            if mask is not None:
                sub_mask = mask.expand(sub_pred.shape)
                sub_pred = sub_pred[sub_mask]
                sub_truth = sub_truth[sub_mask]
            mask=(sub_truth>0)
            mse = torch.nn.functional.mse_loss(sub_pred[mask], sub_truth[mask]) + 1e-5
            psnrs.append(10 * torch.log10(max_val * max_val / mse).item())
        return np.mean(psnrs)
    
class SSIMLoss_def(SSIMLoss):
    def forward(self, pred, truth):
        return 1-ssim(pred,truth,data_range=(truth.max()-truth.min()))
    
class MSSSIMLoss_def(MultiScaleSSIMLoss):
    def forward(self, pred, truth):
        v=ms_ssim(pred,truth,data_range=(truth.max()-truth.min()),size_average=True,win_size=9)
        return 1-v

class L1Loss_mre(torch.nn.L1Loss):
    def forward(self, pred, truth):
        mask=truth>0
        return torch.nn.functional.l1_loss(pred[mask], truth[mask])
    
class MSELoss_mre(torch.nn.MSELoss):
    def forward(self, pred, truth):
        mask=truth>0
        return torch.nn.functional.mse_loss(pred[mask], truth[mask])


import torch
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        feature_layers=[0, 1, 2, 3]
        style_layers=[]
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.mse_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.mse_loss(gram_x, gram_y)
        return loss


class MSELossWeighted(torch.nn.MSELoss):
    def __init__(self, shape=None, weights=None, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        if weights is not None:
            weights = torch.tensor(weights).float().reshape(shape)
            self.register_buffer('weights', weights)
        else:
            self.weights = None

    def forward(self, pred, target):
        pred = self.weights * pred
        target = self.weights * target
        return super().forward(pred, target)

    def extra_repr(self):
        weights = self.weights.squeeze().cpu().tolist()
        return f'weights={weights}, shape={self.shape}'

class Meter:
    def __init__(self):
        self.reset()

    def reset(self):
        self._value = None
        self._sum = 0
        self._num = 0
        self._avg = None

    @property
    def avg(self):
        #print('avg:: ' , self._avg)
        return self._avg

    @property
    def current(self):
        return self._value

    def update(self, value):
        self._value = value.item() if isinstance(value, torch.Tensor) else value
        self._sum += self._value
        self._num += 1
        self._avg = self._sum / self._num


class Meters:
    def __init__(self, formats):
        self._meters = defaultdict(Meter)
        self.formats = formats

    def reset(self):
        for v in self._meters.values():
            v.reset()

    def update(self, key, value):
        self._meters[key].update(value)

    @property
    def avg(self):
        return {k: v.avg for k, v in self._meters.items()}

    @property
    def avg_str(self):
        return {k: f'{v:{self.formats[k]}}' for k, v in self.avg.items()}

    @property
    def current(self):
        return {k: v.current for k, v in self._meters.items()}


class MetricWrapper(torch.nn.Module):   # PSNR
    def __init__(self, cls_name, input_keys, **kwargs):
        super().__init__()
        self.func = globals()[cls_name](**kwargs).cuda()
        self.input_keys = input_keys
    def forward(self, data):
        #print('222222222',self.func)
        
        #print(*[data[k] for k in self.input_keys])
        # if "Loss" in str(self.func):
        #     curr_loss=torch.zeros(1,data['pred_truth'].shape[0],requires_grad=True)
        #     for i in range(data['pred_truth'].shape[0]):
        #         curr_loss[i]=self.func(*[data[k][i,:,:,:] for k in self.input_keys])
        #     return torch.mean(curr_loss), torch.std(curr_loss)
        # else:
        #     return self.func(*[data[k] for k in self.input_keys])
        return self.func(*[data[k] for k in self.input_keys])

    def extra_repr(self):
        return f'input_keys={self.input_keys}'
        
class MetricWrapper_cross(torch.nn.Module):
    def __init__(self, cls_name, input_keys, **kwargs):
        super().__init__()
        self.func = globals()[cls_name](**kwargs).cuda()
        self.input_keys = input_keys

    def forward(self, data):
        
        
        factor = 50     #factor is f
        #print('1111111',self.func)
        
        #print(*[data[k] for k in self.input_keys])
        # for mask
        # if "Loss" in str(self.func):
        #     mask_keys=['pred','mask']
        #     curr_loss=[]
        #     for i in range(data['pred'].shape[0]):
        #         curr_loss.append(self.func(*[torch.sigmoid(factor*(2*data[k][i,:,:,:]-1)) if k=='pred' else data[k][i,:,:,:].type(torch.cuda.FloatTensor) for k in mask_keys]))
        #     curr_loss=torch.tensor(curr_loss,requires_grad=True)
        #     return torch.mean(curr_loss), torch.std(curr_loss)
        # else:
        #     return self.func(*[torch.sigmoid(factor*(2*data[k]-1)) if k=='pred' else data[k].type(torch.cuda.FloatTensor) for k in self.input_keys])
        # return self.func(*[torch.sigmoid(factor*(2*data[k]-1)) if k=='pred' else data[k].type(torch.cuda.FloatTensor) for k in self.input_keys])
        return self.func(*[torch.sigmoid(factor*data[k]) for k in self.input_keys])

    def extra_repr(self):
        return f'input_keys={self.input_keys}'

class LossWrapper(MetricWrapper):
    def __init__(self, cls_name, input_keys, weight, **kwargs):
        super().__init__(cls_name, input_keys, **kwargs)
        self.weight = weight

    def extra_repr(self):
        return ', '.join([super().extra_repr(), f'weight={self.weight}'])

class LossWrapper_cross(MetricWrapper_cross):
    def __init__(self, cls_name, input_keys, weight, **kwargs):
        #print(type(cls_name))
        super().__init__('CrossEntropyLoss', input_keys, **kwargs)
        self.weight = weight

    def extra_repr(self):
        return ', '.join([super().extra_repr(), f'weight={self.weight}'])
        
class Metrics(torch.nn.Module):     # PSNR
    def __init__(self, metrics, meters):
        super().__init__()
        self.metrics = metrics
        self.meters = meters

    @classmethod
    def from_config(cls, configs, meters):
        metrics = torch.nn.ModuleDict()
        for key, config in configs.items():
            metrics[key] = MetricWrapper(
                config.type, config.input_keys, **config.kwargs)
        return cls(metrics, meters)

    def forward(self, data):
        result = dict()
        for k, metric in self.metrics.items():
            result[k] = metric(data)
            self.meters.update(k, result[k])


class Losses(torch.nn.Module):
    def __init__(self, losses, meters):
        super().__init__()
        self.losses = losses
        self.meters = meters

    @classmethod
    def from_config(cls, configs, meters):
        losses = torch.nn.ModuleDict()
        for key, config in configs.items():
            losses[key] = LossWrapper(
                config.type, config.input_keys, config.weight, **config.kwargs)
        return cls(losses, meters)

    def forward(self, data):
        total_loss = 0
        result = dict()
        for k, loss_func in self.losses.items():
            #print('\n')

            #print('loss_func')
 
            # data: input, pred_truth, pred
            result[k] = loss_func(data)     #k=l2

            self.meters.update(k, result[k])
            total_loss += loss_func.weight * result[k]
        self.meters.update('total_loss', total_loss)
        return total_loss
    
    
class Losses_cross(torch.nn.Module):
    def __init__(self, losses, meters):
        super().__init__()
        self.losses = losses
        self.meters = meters

    @classmethod
    def from_config(cls, configs, meters):
        losses = torch.nn.ModuleDict()
        for key, config in configs.items():
            losses[key] = LossWrapper_cross(
                config.type, config.input_keys, config.weight, **config.kwargs)
        #a = cls(losses, meters)
        #print(losses.items())
        return cls(losses, meters)

    def forward(self, data):
        total_loss = 0
        result = dict()
        
        factor = 50
        

        for k, loss_func in self.losses.items():
            #print('\n')

            #print (loss_func)
            #data['pred_truth'].data = torch.sigmoid(factor*data['pred_truth'].data)
            #data['pred'].data = torch.sigmoid(factor*data['pred'].data)
            #print('loss_func11112312312312312312')
            #print (self.losses.items)
            #print('data_pred')
            #print(data['pred_truth'].data)
            #print('\n')
            #print('data')
            #print(data['pred'].data)

            result[k] = loss_func(data)
            self.meters.update(k, result[k])
            total_loss += result[k]
        self.meters.update('total_loss', total_loss)
        return total_loss
