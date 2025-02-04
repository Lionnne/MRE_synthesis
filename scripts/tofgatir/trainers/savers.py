import torch
import numpy as np
import imageio
import pandas as pd
from threading import Thread
from pathlib import Path
from queue import SimpleQueue

from ..utils import NamedData


class SaveJpg:
    """Saves an image as a .jpg file. Assume the image is in [0, 1].

    """
    def __call__(self, named_data):
        filename = Path(named_data.name).with_suffix('.jpg')
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        image = named_data.data
        image = (image * 255).astype(np.uint8).T
        imageio.imsave(filename, image)


class SaveJpgNorm:
    """Saves an image as a .jpg file. Assume the image is in [0, 1].

    """
    def __call__(self, named_data):
        filename = Path(named_data.name).with_suffix('.jpg')
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        image = named_data.data
        image = image / image.max() if image.max()>1 else image
        image = (image * 255).astype(np.uint8).T
        imageio.imsave(filename, image)


class SaveJpg3d:
    """Saves an image as a .jpg file. Assume the image is in [0, 1].

    """
    def __call__(self, named_data):
        filename = Path(named_data.name).with_suffix('.jpg')
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        image = named_data.data
        ind = image.shape[-1] // 2
        image = image[..., ind]
        if image.max()>1:
            print(image.max())
            image=image/image.max()
        image = (image * 255).astype(np.uint8).T
        imageio.imsave(filename, image)


class SaveJpgNorm3d:
    """Saves an image as a .jpg file. Assume the image is in [0, 1].

    """
    def __call__(self, named_data):
        filename = Path(named_data.name).with_suffix('.jpg')
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        image = named_data.data
        ind = image.shape[-1] // 2
        image = image[..., ind]
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8).T
        imageio.imsave(filename, image)


class SaveCsv:
    """Saves params as a .csv file.

    """
    def __call__(self, named_data):
        filename = Path(named_data.name).with_suffix('.csv')
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(named_data.data)
        df.to_csv(filename, header=False, index=False)


class SaveData:
    def __init__(self, save_mapping):
        self._save_data = {k: globals()[v]() for k, v in save_mapping.items()}
    def __call__(self, key, named_data):
        self._save_data.get(key, SaveJpgNorm())(named_data)


class DataSavingThread(Thread):
    """Saves data in a thread.

    Args:
        save_data (SaveData): Save images to files.
        queue (queue.Queue): The queue to transfer the data.

    """
    def __init__(self, save_data, queue):
        super().__init__()
        self.save_data = save_data
        self.queue = queue

    def run(self):
        while True:
            data = self.queue.get() # almost zeros
            if data is None:
                break
            self.save_data(*data)   # named_data


class DataSaver:
    def __init__(
        self, dirname, save_mapping, total_num_epochs=None, stack_size=1
    ):
        self.dirname = dirname
        self.total_num_epochs = total_num_epochs
        self.stack_size = stack_size
        self._queue = SimpleQueue()
        self._thread = DataSavingThread(SaveData(save_mapping), self._queue)

    def start(self):
        self._thread.start()

    def close(self):
        self._queue.put(None)
        self._thread.join()

    def save(self, data, epoch_ind):
        for k, v in data.items():
            if k == 'params':
                results = self._parse_params(v)
            else:
                results = self._parse_images(k, v)
            for result in results:
                self._save(k, result, epoch_ind + 1)

    def _parse_params(self, value):
        data = value.data.detach().cpu().numpy()
        name = ['_'.join(['params', n]) for n in value.name]
        result = [NamedData(name=name, data=data)]
        return result

    def _parse_images(self, key, value):
        results = list()
        data = value.data.detach().cpu().numpy()
        names = list(zip(*[self._parse_name(key, name) for name in value.name]))
        if key == 'input' and self.stack_size is not None:
            slice_ind = self.stack_size // 2
            slice_range = range(slice_ind, value.data.shape[1], self.stack_size)
        else:
            slice_range = range(value.data.shape[1])
        for i, s in enumerate(slice_range):
            results.append(NamedData(name=names[i], data=data[:, s, ...]))  #start and end
        return results

    def _parse_name(self, key, name):
        names = name.split('_')
        prefix = '_'.join(names[:3])
        suffix = '_'.join(names[4:])
        modalities = names[3].split('-')
        results = ['_'.join([key, prefix, m, suffix]) for m in modalities]
        return results

    def _save(self, key, named_data, epoch):
        subdir = self._get_prefix(epoch, 'epoch', self.total_num_epochs)
        for i, name in enumerate(named_data.name):
            prefix = self._get_prefix(i, 'sample', len(named_data.name))
            filename = Path(self.dirname, subdir, '_'.join([prefix, name]))
            result = NamedData(name=filename, data=named_data.data[i])
            self._queue.put((key, result))

    def _get_prefix(self, num, prefix, total_num):
        num_digits = 8
        if total_num is not None:
            num_digits = len(str(total_num))
        prefix = f'{prefix}-{num:0{num_digits}d}'
        return prefix


class CheckpointSaver:
    """Saves models and optimizers periodically.

    """
    def __init__(
        self, model, optim, dirname, total_num_epochs=None, scheduler=None
    ):
        self.model = model
        self.optim = optim
        self.dirname = dirname
        self.total_num_epochs = total_num_epochs
        self.scheduler = scheduler

    def save(self, epoch, is_best=False, **kwargs):
        epoch += 1
        if is_best:
            filename = Path(self.dirname, f'ckpt_best_{epoch:0{6}d}.pt')
        else:
            filename = self._get_filename(epoch)
        to_save = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
        }
        if self.scheduler is not None:
            to_save['scheduler_state_dict'] = self.scheduler.state_dict()
        to_save.update(kwargs)
        torch.save(to_save, filename)

    def _get_filename(self, epoch):
        num_digits = 8
        if self.total_num_epochs is not None:
            num_digits = len(str(self.total_num_epochs))
        filename = f'ckpt_{epoch:0{num_digits}d}.pt'
        filename = Path(self.dirname, filename)
        return filename
