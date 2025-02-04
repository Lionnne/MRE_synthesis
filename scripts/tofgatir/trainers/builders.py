import random
import logging
import re
import numpy as np
import nibabel as nib
import yaml
from datetime import datetime
from easydict import EasyDict
from pathlib import Path
import torch.optim

from .. import models
from ..losses import Metrics, Losses, Meters, Losses_cross
from .trainers import Trainer, Validator, Evaluator, Evaluator3d, Evaluator3dPatches
from .savers import DataSaver, CheckpointSaver
from ..utils import LOGGER_NAME, dump_easydict_to_yml
from ..data import DataLoaderFactory
from ..models.networks import AttU_Net
from ..models.networks import LinearRegressionModel


class _Builder:
    def __init__(self, args):
        self.args = args
        self.config = self._parse_config(args.config)
        self._get_logdir()
        self._set_seed()
        torch.cuda.set_device(args.gpu)
        config_fn = Path(self.config.experiment.logdir, 'config.yml')
        dump_easydict_to_yml(config_fn, self.config)

    def _get_logdir(self):
        suffix = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        logdir = Path(self.config.experiment.logdir, self.config.experiment.id)
        logdir = '_'.join([str(logdir), suffix])
        self.config.experiment.logdir = logdir
        Path(self.config.experiment.logdir).mkdir(parents=True, exist_ok=True)

    def _parse_config(self, config):
        with open(config) as f:
            config = yaml.safe_load(f)
        config = EasyDict(config)
        return config

    def _set_seed(self):
        if hasattr(self.config.experiment, 'seed'):
            np.random.seed(self.config.experiment.seed)
            random.seed(self.config.experiment.seed)
            torch.manual_seed(self.config.experiment.seed)

    def _create_logger(self, prefix):
        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(logging.INFO)
        logname = Path(self.config.experiment.logdir, prefix)
        logname = logname.with_suffix('.log')
        handler = logging.FileHandler(logname)
        formatter = logging.Formatter('%(asctime)s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger


class TrainerBuilder(_Builder):
    def build(self):
        logger = self._create_logger('train')

        loader_factory = DataLoaderFactory(self.config.data)
        train_loader = loader_factory.create_dataloader('train')
        valid_loader = loader_factory.create_dataloader('valid')
        logger.info('Training data: ' + train_loader.dataset.__str__())
        logger.info('Validation data: ' + valid_loader.dataset.__str__())

        Model = getattr(models, self.config.model.type)
        model = Model(**self.config.model.kwargs).cuda()
        # model = AttU_Net().cuda()
        # model = LinearRegressionModel().cuda()  #30-50
        Optim = getattr(torch.optim, self.config.optim.type)
        optim = Optim(model.parameters(), **self.config.optim.kwargs)
        logger.info(model.__str__())
        logger.info(optim.__str__())
        scheduler = None
        if 'lr_scheduler' in self.config:
            scheduler_cls = getattr(
                torch.optim.lr_scheduler, self.config.lr_scheduler.type)
            scheduler = scheduler_cls(
                optim, epochs=self.config.experiment.num_epochs,
                steps_per_epoch=len(train_loader),
                **self.config.lr_scheduler.kwargs
            )
            logger.info(scheduler.__str__())

        start_epoch = 0
        if self.config.experiment.checkpoint is not None:
            ckpt = torch.load(self.config.experiment.checkpoint)
            model.load_state_dict(ckpt['model_state_dict'])
            optim.load_state_dict(ckpt['optim_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch']
            logger.info(f'Load checkpoint. The starting epoch is {start_epoch}.')

        stack_size = self.config.data.image_reader.kwargs.get('stack_size', None)
        valid_meters = Meters(self.config.log_formats)
        valid_calc_loss = Losses.from_config(self.config.losses, valid_meters)
        valid_calc_loss1 = Losses_cross.from_config(self.config.losses, valid_meters)
        valid_calc_metrics = Metrics.from_config(
            self.config.metrics, valid_meters)
        valid_dirname = Path(self.config.experiment.logdir, 'valid')
        valid_image_saver = DataSaver(
            valid_dirname, self.config.data.saving_types,
            self.config.experiment.num_epochs, stack_size=stack_size
        )
        ckpt_saver = CheckpointSaver(
            model, optim, self.config.experiment.logdir, scheduler=scheduler)
        validator = Validator(
            self.config.experiment, valid_loader, model, valid_calc_loss,
            valid_calc_metrics, image_saver=valid_image_saver, logger=logger,
            ckpt_saver=ckpt_saver,train_calc_loss1 = valid_calc_loss1
        )
        logger.info(valid_calc_loss.__str__())
        logger.info(valid_calc_metrics.__str__())

        train_meters = Meters(self.config.log_formats)
        train_calc_loss = Losses.from_config(self.config.losses, train_meters)
        
        train_calc_loss1 = Losses_cross.from_config(self.config.losses, train_meters)
        
        train_calc_metrics = Metrics.from_config(
            self.config.metrics, train_meters)
        train_dirname = Path(self.config.experiment.logdir, 'train')
        train_image_saver = DataSaver(
            train_dirname, self.config.data.saving_types,
            self.config.experiment.num_epochs, stack_size=stack_size
        )
        trainer = Trainer(
            self.config.experiment, train_loader, model, optim, train_calc_loss,
            train_calc_metrics, logger=logger, validator=validator,
            image_saver=train_image_saver, ckpt_saver=ckpt_saver,
            start_epoch=start_epoch, scheduler=scheduler,
            train_calc_loss1 = train_calc_loss1
        )
        return trainer


class EvaluatorBuilder(_Builder):
    def __init__(self, args):
        self.args = args
        self.config = self._parse_config(args.config)
        torch.cuda.set_device(args.gpu)

    def build(self, model=None):
        
        loader_factory = DataLoaderFactory(self.config.data)
        loader = loader_factory.create_eval_dataloader(
            self.args.filenames, self.args.params)
        if model is None:
            Model = getattr(models, self.config.model.type)
            model = Model(**self.config.model.kwargs)
            ckpt = torch.load(
                self.args.checkpoint, map_location=f'cuda:{self.args.gpu}')
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.cuda()
            print(f'Load checkpoint {self.args.checkpoint}, '
                  f'best epoch {ckpt["epoch"]}.')
        a = self.args.filenames[0]
        a = a.replace('/test_memmap', '/test')
        b = re.sub(r'_data.dat$', '.nii.gz', a)
        # print(b)
        orig_shape = nib.load(self.args.filenames[0]).shape
        test_config = loader_factory._get_config('test')
        if 'Patches' in test_config.image_reader.type:
            border = self.config.data.get('border', 0)
            evaluator = Evaluator3dPatches(loader, model, orig_shape, border=border)
        elif 'Slices' in test_config.image_reader.type:
            evaluator = Evaluator(loader, model, orig_shape)
        else:
            evaluator = Evaluator3d(loader, model, orig_shape)
        return evaluator, model
