import numpy as np
from cv2 import resize, INTER_CUBIC
import random

from ..utils import NamedData, padcrop


class FlipLR:
    def __str__(self):
        return f'{self.__class__.__name__}()'

    def __call__(self, *images):
        results = images
        need_to_flip = random.uniform(0, 1) < 0.5
        need_to_flip = ('axis-0' not in images[0].name) and need_to_flip
        if need_to_flip:
            results = list()
            for image in images:
                assert image.data.ndim == 3
                data = np.flip(image.data, axis=1).copy()
                name = '_'.join([image.name, 'flip'])
                results.append(NamedData(name=name, data=data))
        return results


class Scale:
    def __init__(self, scale=1.2):
        self.scale = scale
        assert self.scale >= 1

    def __str__(self):
        name = self.__class__.__name__
        return f'{name}(scale={self.scale})'

    def __call__(self, *images):
        results = images    # NameData
        need_to_scale = random.uniform(0, 1) < 0.5
        if need_to_scale:
            results = list()
            scales = self._sample_scales()  # 2 scale para bet 1,1.2
            for image in images:    # NameData
                assert image.data.ndim == 3
                data = self._scale_images(image.data, scales)
                name = self._get_name(image.name, scales)
                results.append(NamedData(name=name, data=data))
        return results

    def _sample_scales(self):
        flip_signs = random.choices([False, True], k=2)
        scales = [random.uniform(1, self.scale) for _ in range(2)]
        result = [1 / s if f else s for f, s in zip(flip_signs, scales)]
        return result

    def _get_name(self, image_name, dxy):
        dxy = [('%.2f' % d).replace('.', 'p') for d in dxy]
        dxy = '-'.join(dxy)
        name = '_'.join([image_name, 'scale-%s' % dxy])
        return name

    def _scale_image(self, data, scales):
        target_shape = [int(round(data.shape[1] * scales[1])),
                        int(round(data.shape[0] * scales[0]))]
        result = resize(data, target_shape)
        result = padcrop(result, data.shape, use_channels=False)
        return result

    def _scale_images(self, data, scales):
        results = list()
        for image in data:  # 160,160
            d = self._scale_image(image, scales)
            results.append(d)
        return np.stack(results, axis=0)


class Compose:
    """Composes several transforms together.

    """
    def __init__(self, transforms):
        self.transforms = transforms

    @classmethod
    def from_config(cls, config):
        transforms = list()
        for trans_config in config:
            Trans = globals()[trans_config.type]
            kwargs = trans_config.get('kwargs', {})
            transforms.append(Trans(**kwargs))
        return cls(transforms)

    def __str__(self):
        result = [f'{self.__class__.__name__}(']
        for trans in self.transforms:
            result.append(f'  {trans.__str__()}')
        result.append(')')
        if len(result) > 2:
            result = '\n'.join(result)
        else:
            result = ''.join(result)
        return result

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args