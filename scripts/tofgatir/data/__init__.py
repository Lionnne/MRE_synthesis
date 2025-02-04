import numpy as np
from torch.utils.data import DataLoader
from copy import deepcopy
from collections import OrderedDict

from .dataset import GroupImages, Dataset, SubjectData, ParamReader
from .transforms import Compose


class DataLoaderFactory:
    def __init__(self, config):
        self.config = config

    def create_dataloader(self, key):
        config = self._get_config(key)
        group_images = GroupImages(
            config.dirname, config.loading_order, ext=config.ext)
        filenames = group_images.group()    # T1, T2,50Hz, OrderedDict()
        
        transform = Compose.from_config(config.transforms)
        params = None
        if 'params' in config:
            params = ParamReader(config.params)
        subjects = list()   # all train data (160,160,80)
        for name, fns in filenames.items():     #name:U01xxx, fns:T1/T2/50Hz
            sub_params = None
            if params is not None:
                all_keys = config.loading_order
                truth_ind = config.datadict_indices['pred_truth']
                truth_keys = [all_keys[i] for i in truth_ind]
                sub_params = params.get_sub_params(name, truth_keys)
            subjects.append(SubjectData(
                name, fns, params=sub_params, transform=transform,
                reader_type=config.image_reader.type,
                reader_kwargs=config.image_reader.kwargs,
                negate_ind=config.negate_ind        # what index?
            ))  # subjects -1,1
        shuffle_once = key == 'valid'
        border = self.config.get('border', 0)
      
        dataset = Dataset(
            subjects, config.datadict_indices, config.center_slice_keys,
            data_dtypes=config.data_dtypes, apply_mask=config.apply_mask,
            shuffle_once=shuffle_once, border=border,
            mask_rand=config.get('mask_rand', False),
            num=config.get('mask_rand_num', None),
        )
        dataloader = DataLoader(
            dataset, batch_size=config.batch_size,
            num_workers=config.num_workers, shuffle=config.shuffle,
            pin_memory=True, drop_last=True
        )
        return dataloader

    def _get_config(self, key):
        all_keys = {'train', 'valid', 'test'}
        config = deepcopy(self.config)
        for other_key in all_keys - {key}:
            config.pop(other_key)
        key_config = config.pop(key)
        config.update(key_config)
        return config

    def create_eval_dataloader(self, filenames, params):
        config = self._get_config('test')
        if params is not None:
            params = [[float(n) for n in nums.split(',')] for nums in params]
            params = np.array(params).astype(np.float32)
        filenames = {k: v for k, v in zip(config.loading_order, filenames)}
        subject_data = SubjectData(
            'subj', OrderedDict(filenames), params=params,
            reader_type=config.image_reader.type,
            reader_kwargs=config.image_reader.kwargs,
            negate_ind=config.negate_ind
        )
        dataset = Dataset(
            [subject_data], config.datadict_indices, config.center_slice_keys,
            apply_mask=config.apply_mask, data_dtypes=config.data_dtypes,
            border=self.config.get('border', 0)
        )
        dataloader = DataLoader(
            dataset, batch_size=config.batch_size,
            num_workers=config.num_workers, shuffle=False,
            pin_memory=True, drop_last=False
        )
        return dataloader
