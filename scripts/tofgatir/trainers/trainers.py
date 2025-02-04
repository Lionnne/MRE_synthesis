import numpy as np
import torch
from tqdm import trange
from copy import deepcopy
from collections import defaultdict
from improc3d import padcrop3d
from piq import ssim, SSIMLoss, MultiScaleSSIMLoss

from ..utils import NamedData, move_nameddata_dict_to_cuda, padcrop


class _Base:
    def start(self):
        if self.image_saver is not None:
            self.image_saver.start()

    def run(self):
        raise NotImplementedError

    def close(self):
        if self.image_saver is not None:
            self.image_saver.close()

    def _log(self, epoch_ind, prefix='train'):
        prefix = '[' + prefix.upper() + ']'
        metrics = self.calc_loss.meters.avg_str
        metrics = [f'{k}: {v}' for k, v in metrics.items()]
        message = [prefix, f'epoch: {epoch_ind}'] + metrics
        self.logger.info(', '.join(message))

    def _predict(self, data):
        # ## Normalization of data
        # data['input'].data = preprocessing.MaxAbsScaler().fit_transform(data['input'].data)
        # data['pred_truth'].data = preprocessing.MaxAbsScaler().fit_transform(data['pred_truth'].data)
        data = move_nameddata_dict_to_cuda(data)
        params = data['params'].data if 'params' in data else None
        pred = self.model(data['input'].data, params=params)    # the last layer
        # pred = self.model(data['input'].data)   # Atten

        for k, v in pred.items():
            # print('k : ')
            # print(k)
            # print('v : ')
            # print(v)

            # back to original intensity range?
            data[k] = NamedData(name=data[f'{k}_truth'].name, data=v)   #in,truth,mask,pred
            # data：input, pred_truth, pred
            # print(data[k])


        loss = self.calc_loss({k: v.data for k, v in data.items()}) # l2 loss
        #print('\n')
        #print(loss)

       
        # # compute the second loss for mask
        # loss2 = self.train_calc_loss1({k: v.data for k, v in data.items()})     # loss2=0?
        torch.set_printoptions(precision=10)
        #print(loss2)
        
        
        # s1 = 220    # why?
        # s2 = 0.999
        
        # loss3 = (1/(2*(s1**2))) * loss + (1/(2*(s2**2))) * loss2 + torch.tensor(np.log(s1*s2))
        
        
        with torch.no_grad():
            self.calc_metrics({k: v.data for k, v in data.items()})
        return loss.cuda()

    def _need_to_save_images(self, epoch_ind):
        return self.image_saver is not None \
            and (epoch_ind + 1) % self.config.save_image_step == 0


class Trainer(_Base):
    def __init__(
        self, config, loader, model, optim, calc_loss, calc_metrics,
        validator=None, image_saver=None, ckpt_saver=None, logger=None,
        start_epoch=0, scheduler=None,train_calc_loss1 = None
    ):
        self.config = config
        self.loader = loader
        self.model = model
        #print(self.model)
        self.optim = optim
        self.calc_loss = calc_loss
        self.calc_metrics = calc_metrics
        self.validator = validator
        self.image_saver = image_saver
        self.ckpt_saver = ckpt_saver
        self.logger = logger
        self.start_epoch = start_epoch
        self.scheduler = scheduler
        self.train_calc_loss1 = train_calc_loss1

        self._batch_pbar = trange(
            len(self.loader), desc='batch', ncols=60, position=0)
        self._epoch_pbar = trange(
            self.config.num_epochs, desc='epoch', ncols=60, position=1)

    def start(self):
        super().start()
        if self.validator is not None:
            self.validator.start()

    def run(self):
        self.model.train()
        self._epoch_pbar.update(self.start_epoch)
        self._epoch_pbar.refresh()
        for epoch_ind in range(self.start_epoch, self.config.num_epochs):
            self._batch_pbar.reset()
            self.calc_loss.meters.reset()
            for data in self.loader:    # input pred_truth, mask
                self.optim.zero_grad()
                loss = self._predict(data)
                loss.backward()
                self.optim.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                    last_lr = self.scheduler.get_last_lr()[0]
                    self.calc_loss.meters.update('lr', last_lr)
                self._batch_pbar.update()
                self._batch_pbar.refresh()
            self._epoch_pbar.update()
            self._epoch_pbar.refresh()
            if self._need_to_print(epoch_ind):
                self._log(epoch_ind, 'train')
            if self._need_to_save_images(epoch_ind):
                self.image_saver.save(data, epoch_ind)
            if self._need_to_validate(epoch_ind):
                self.validator.run(epoch_ind)
            if self._need_to_save_ckpt(epoch_ind):
                
                path = self.ckpt_saver.dirname
                with open( path + '/training_loss.txt', "a") as f:
                    
                    f.write(str(loss.item()) + '\n')
                self.ckpt_saver.save(epoch_ind)
            torch.cuda.empty_cache()

    def _need_to_print(self, epoch_ind):
        return self.logger is not None \
            and (epoch_ind + 1) % self.config.print_step == 0

    def _need_to_validate(self, epoch_ind):
        return self.validator is not None \
            and (epoch_ind + 1) % self.config.valid_step == 0

    def _need_to_save_ckpt(self, epoch_ind):
        return self.ckpt_saver is not None \
            and (epoch_ind + 1) % self.config.save_ckpt_step == 0

    def close(self):
        super().close()
        if self.validator is not None:
            self.validator.close()
        self._batch_pbar.close()
        self._epoch_pbar.close()


class Validator(_Base):
    def __init__(
        self, config, loader, model, calc_loss, calc_metrics,
        image_saver=None, logger=None, ckpt_saver=None, train_calc_loss1 = None
    ):
        self.config = config
        self.loader = loader
        self.model = model
        self.calc_loss = calc_loss
        self.calc_metrics = calc_metrics
        self.image_saver = image_saver
        self.ckpt_saver = ckpt_saver
        self.logger = logger
        self.train_calc_loss1 = train_calc_loss1
        self._min_loss = float('inf')
        self._pbar = trange(
            len(self.loader), desc='valid', ncols=60, position=2)
        self.total_loss3 = 0
        self.average_loss3= 0
    def run(self, epoch_ind):
        self.model.eval()
        self._pbar.reset()
        self.calc_loss.meters.reset()
    
        with torch.no_grad():
            for data in self.loader:
                self.total_loss3 += self._predict(data)
                self._pbar.update()
                self._pbar.refresh()
        self._log(epoch_ind, 'valid')
        self.average_loss3 = self.total_loss3 / len(self.loader)
        if self.image_saver is not None:
            self.image_saver.save(data, epoch_ind)
        if self._need_to_save_ckpt(epoch_ind):
            self.ckpt_saver.save(
                epoch_ind, is_best=True, min_loss=self._min_loss)
        self.model.train()

    def _need_to_save_ckpt(self, epoch_ind):
        #print('\n')
        # avg_loss = self.calc_loss.meters.avg['total_loss']
        #print('121121212112',avg_loss)
        #print(self.calc_loss.meters.avg)
       
        #print('33434343434343',self.average_loss3)
        #print(self.train_calc_loss1.meters.avg)
        if self.average_loss3< self._min_loss:
            self._min_loss = self.average_loss3
            if epoch_ind >= self.config.best_ckpt_idle_epochs \
                    and self.ckpt_saver is not None:
                return True
        return False

    def close(self):
        super().close()
        self._pbar.close()


class Evaluator(_Base):
    def __init__(self, loader, model, orig_shape, *args, **kwargs):
        self.loader = loader
        self.model = model
        self.orig_shape = orig_shape
        self._pbar = trange(
            len(self.loader), desc='test', ncols=60, position=2)

    def run(self):
        self.model.eval()
        outputs = defaultdict(list)
        names = defaultdict(list)
        with torch.no_grad():
            for data in self.loader:
                data = move_nameddata_dict_to_cuda(data)
                params = data['params'].data if 'params' in data else None
                pred = self.model(data['input'].data, params=params)
                for k, v in pred.items():
                    outputs[k].append(v.detach().cpu())
                    names[k].append(data['input'].name)
                self._pbar.update()
                self._pbar.refresh()
        return self._postproc(outputs, names)

    def _postproc(self, outputs, *args):
        for k in outputs.keys():
            outputs[k] = torch.cat(outputs[k], dim=0).numpy()   #ori:.abs().numpy()
        stack_size = self.loader.dataset.subjects[0].reader_kwargs.stack_size
        with_stack_shape = [s - stack_size + 1 for s in self.orig_shape]
        stops = np.cumsum(with_stack_shape).tolist()
        starts = [0] + stops[:2]
        result = defaultdict(dict)
        for k, output in outputs.items():
            for axis, (start, stop) in enumerate(zip(starts, stops)):
                result[f'axis{axis}'][k] = self._reshape_data(
                    output[start : stop, ...], axis, stack_size)
            result['median'][k] = np.median(
                [result[f'axis{i}'][k] for i in range(3)], axis=0)
        return result

    def _reshape_data(self, data, axis, stack_size):
        padding = [(0, 0)] * data.ndim
        padding[0] = (stack_size//2,) * 2
        data = np.pad(data, padding)
        data_shape = list(self.orig_shape)
        data_shape.pop(axis)
        data = padcrop(data, data_shape, use_channels=True)
        if axis in [0, 1]: # top bottom swap
            data = np.flip(data, -1)
        data = np.moveaxis(data, 0, axis + 1)
        return data

    def close(self):
        self._pbar.close()


class Evaluator3dPatches(Evaluator):
    def __init__(self, loader, model, orig_shape, border=0):
        super().__init__(loader, model, orig_shape)
        self.border = border

    def _postproc(self, outputs, names):
        result = {'3d': dict()}
        for k, output in outputs.items():
            result['3d'][k] = np.zeros((output[0].shape[1],) + self.orig_shape)
            name = names[k]
            for n, o in zip(name, output):
                for nn, oo in zip(n, o):
                    ind = [int(i) for i in nn.split('_')[2].split('-')[1:]]
                    ind = [i + self.border for i in ind]
                    result_ind = tuple(slice(i, i + s) for i, s in zip(ind, oo.shape[1:]))
                    patch_ind = tuple(slice(0, s - i) for s, i in zip(self.orig_shape, ind))
                    result['3d'][k][(...,) + result_ind] = oo[(...,) + patch_ind]
        return result


class Evaluator3d(Evaluator):
    def _postproc(self, outputs, names):
        result = {'3d': dict()}
        for k, output in outputs.items():
            assert len(output) == 1 and len(output[0]) == 1
            result['3d'][k] = padcrop3d(output[0][0].numpy(), self.orig_shape, False)
        return result
