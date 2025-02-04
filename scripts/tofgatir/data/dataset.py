import re
import random
import nibabel as nib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as _Dataset
from pathlib import Path
from collections import OrderedDict
from improc3d import padcrop3d

from ..utils import padcrop, NamedData, map_data


class ImageSlicesPreLoad:
    def __init__(self, filename, name, stack_size=1, sign=1, target_shape=None):
        self.data = nib.load(filename)
        self.name = name
        self.target_shape = target_shape
        self.stack_size = stack_size
        self.sign = sign
        assert self.stack_size % 2 == 1
        self._num_cumsum = np.cumsum(self.shape)

    @property
    def shape(self):
        return tuple(s - self.stack_size + 1 for s in self.data.shape)

    def __getitem__(self, index):
        self._check_index(index)
        axis, slice_ind = self._split_index(index)
        image_slice = self.extract_slice(axis, slice_ind)
        return image_slice  # NameData

    def _split_index(self, index):
        axis = np.searchsorted(self._num_cumsum, index, side='right')
        offset = 0 if axis == 0 else self._num_cumsum[axis - 1]
        index = index - offset
        return axis, index

    def extract_slice(self, axis, index):
        indexing = self._calc_indexing(axis, index)
        image_slice = self._extract_slice(indexing) #12后到17 13-17     #-1,1 checked 160,5,80
        image_slice = np.moveaxis(image_slice, axis, 0) # axis stack be the first->5,160,80
        if axis in [0, 1]: # top bottom swap    # why swap
            image_slice = np.flip(image_slice, -1)
        if self.target_shape:
            image_slice = padcrop(image_slice, self.target_shape)
        name = self._get_name(axis, index)
        image_slice = NamedData(name=name, data=image_slice * self.sign)    #5.160，160
        return image_slice

    def _extract_slice(self, indexing):
        if not hasattr(self, '_image_buffer'):
            self._image_buffer = self.data.get_fdata(dtype=np.float32)
        image_slice = self._image_buffer[indexing]
        return image_slice

    def _get_name(self, axis, slice_ind):
        name = [f'axis-{axis}', f'slice-{slice_ind}']
        if self.name:
            name.insert(0, self.name)
        return '_'.join(name)

    def _calc_indexing(self, axis, index):
        result = [slice(None)] * self.data.ndim
        result[axis] = slice(index, index + self.stack_size)    # 向后偏移stack
        return tuple(result)

    def __len__(self):
        return self._num_cumsum[-1]

    def _check_index(self, index):
        assert index >= 0 and index < len(self)


class ImageSlicesDataobj(ImageSlicesPreLoad):
    def _extract_slice(self, indexing):
        return self.data.dataobj[indexing].astype(np.float32)


class ImageSlicesMemmap(ImageSlicesPreLoad):
    def __init__(self, filename, name, stack_size=1, sign=1, target_shape=None):
        #
        dat_fn  = re.sub(r'.nii.gz$' if "gz" in filename else r'.nii', '_data.dat', str(filename))
        shape_fn = re.sub(r'_data\.dat$', '_shape.npy', str(filename))
        dtype_fn = re.sub(r'_data\.dat$', '_dtype.txt', str(filename))
        
        
        #print('fff \n' )
        # dat_fn  = re.sub(r'.nii.gz$', '_data.dat', str(filename))       # what is this for?
        if "test" in dat_fn:
            dat_fn  = dat_fn.replace('Data','Data/Memmap')
            shape_fn = re.sub(r'.nii.gz$' if "gz" in filename else r'.nii', '_shape.npy', str(filename))
            dtype_fn = re.sub(r'.nii.gz$' if "gz" in filename else r'.nii', '_dtype.txt', str(filename))
            shape_fn  = shape_fn.replace('Data','Data/Memmap')
            dtype_fn  = dtype_fn.replace('Data','Data/Memmap')
            filename = dat_fn
        #print(dat_fn)
        
        with open(dtype_fn) as f:
            dtype = f.readline()
        shape = tuple(np.load(shape_fn).tolist())
        self.data = np.memmap(filename, dtype=dtype, mode='r', shape=shape) #(160,160,80)   # -1 for background
        # print(self.data.max())
        # print(self.data.min())
        self.sign = sign

        self.name = name
        self.target_shape = target_shape
        self.stack_size = stack_size
        assert self.stack_size % 2 == 1
        self._num_cumsum = np.cumsum(self.shape)

    def _extract_slice(self, indexing):
        return self.data[indexing].copy()   # (:,:,12)-(:,:,17)


class ImagePatchesMemmap(ImageSlicesMemmap):
    def __init__(self, filename, name, patch_size=32, stride=16, sign=1):
        super().__init__(filename, name, sign=sign)
        self.patch_size = (patch_size,) * 3
        self.stride = (stride,) * 3
        self._coords = self._get_coords()

    def _get_coords(self):
        # stops = [(s - st) // st * st for s, st in zip(self.data.shape, self.stride)]
        indices = [
            np.arange(0, stop, step)
            for stop, step in zip(self.data.shape, self.stride)
        ]
        coords = np.meshgrid(*indices, indexing='ij')
        coords = np.array([c.flatten() for c in coords]).T
        return coords

    @property
    def shape(self):
        return self.data.shape

    def __len__(self):
        return self._coords.shape[0]

    def __getitem__(self, index):
        self._check_index(index)
        coords = self._coords[index, :]
        indexing = tuple(slice(c, c + p) for c, p in zip(coords, self.patch_size))
        patch = np.array(self._extract_patch(indexing))
        padding = [[0, ps - s] for ps, s in zip(self.patch_size, patch.shape)]
        patch = np.pad(patch, padding)
        name = self._get_name(coords)
        patch = NamedData(name=name, data=patch[None, ...] * self.sign)
        return patch

    def _extract_patch(self, indexing):
        return self.data[indexing].copy()

    def _get_name(self, coords):
        name = [f'3d', '-'.join(['ind', *[str(c) for c in coords]])]
        if self.name:
            name.insert(0, self.name)
        return '_'.join(name)


class ImagePatchesPreLoad(ImagePatchesMemmap):
    def __init__(self, filename, name, patch_size=32, stride=16, sign=1):
        self.data = nib.load(filename)
        self.name = name
        self.sign = sign
        self.patch_size = (patch_size,) * 3
        self.stride = (stride,) * 3
        self._coords = self._get_coords()

    def _extract_patch(self, indexing):
        if not hasattr(self, '_image_buffer'):
            self._image_buffer = self.data.get_fdata(dtype=np.float32)
        return self._image_buffer[indexing]


class ImagePreLoad(ImagePatchesPreLoad):
    def __init__(self, filename, name, sign=1, target_shape=(256, 256, 256)):
        self.data = nib.load(filename)
        self.name = name
        self.sign = sign
        self.target_shape = target_shape

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return NamedData(name=self._get_name(), data=self._get_image())

    def _get_image(self):
        if not hasattr(self, '_image_buffer'):
            self._image_buffer = self.data.get_fdata(dtype=np.float32)
            self._image_buffer = padcrop3d(
                self._image_buffer, self.target_shape, False)
            self._image_buffer = self._image_buffer[None, ...] * self.sign
        return self._image_buffer

    def _get_name(self):
        name = ['3d', 'all']
        if self.name:
            name.insert(0, self.name)
        return '_'.join(name)


class _GroupImages:
    def group(self):
        raise NotImplementedError


class GroupImages(_GroupImages):
    def __init__(self, dirname, loading_order=[], ext=r'\.nii(\.gz)*$'):
        self.dirname = dirname
        self.loading_order = loading_order
        self.ext = ext
        self._re_pattern = '(' + '|'.join(self.loading_order) + ')'

    def group(self):
        images = OrderedDict()
        for image_fn in self._find_images():
            names = self._parse_name(image_fn)
            subname, im_type = names[:2]
            if subname not in images:       # add subject to dict
                images[subname] = OrderedDict()
            images[subname][im_type] = str(image_fn)
        return images

    def _find_images(self):
        pattern = self._re_pattern + '.*' + self.ext
        def has_im_type(fn):
            return re.search(pattern, fn.name)
        filenames = Path(self.dirname).iterdir()
        filenames = list(filter(has_im_type, filenames))
        filenames = self._sort_images(filenames)
        return filenames

    def _sort_images(self, filenames):
        keys = {k: v for v, k in enumerate(self.loading_order)}
        def get_sort_key(fn):
            return keys[re.search(self._re_pattern, fn.name).group()]
        filenames = sorted(filenames, key=get_sort_key)
        def get_subj_name(fn):
            return fn.name.split('_')[0]
        filenames = sorted(filenames, key=get_subj_name)
        return filenames

    def _parse_name(self, image_fn):
        return re.sub(self.ext, '', Path(image_fn).name).split('_')


class ParamReader:
    def __init__(self, filename, ndim=2):
        self.params = pd.read_csv(filename, index_col=0)

    def get_sub_params(self, name, keys):
        result_data = list()
        existed_keys = list()
        for key in keys:
            imname = f'{name}_{key}'
            if imname in self.params.index:
                existed_keys.append(key)
                result_data.append(self.params.loc[imname])
        result_data = np.array(result_data).astype(np.float32)
        result_name = f'{name}_{"-".join(existed_keys)}'
        result = NamedData(name=result_name, data=result_data)
        return result


class SubjectData:
    def __init__(
            self, name, filenames, params=None, negate_ind=[], transform=None,
            reader_type='ImageSlicesMemmap', reader_kwargs=dict()
        ):  #filenames:T1,T2,50Hz
        self.name = name    # U01UDEL001
        self.params = params
        self.negate_ind = negate_ind
        self.transform = transform
        self.im_types = list(filenames.keys())  #['T1','T2','50Hz']
        self.reader_kwargs = reader_kwargs  # {'target_shape': [160, 160], 'stack_size': 5}
        self._create_images(filenames, reader_type, reader_kwargs)
        self._check_im_shapes()

    def _create_images(self, filenames, reader_type, reader_kwargs):
        image_slices_cls = globals()[reader_type]   # class variables
        self._images = OrderedDict()
        #print(image_slices_cls)
        for i, (im_type, fn) in enumerate(filenames.items()):
            sign = -1 if i in self.negate_ind else 1
            self._images[im_type] = image_slices_cls(
                fn, self.name, sign=sign, **reader_kwargs
            )   #ImageSlicesMemmap  _num_cumsum([156,312,388])<-(156,156,76)
        # -1,1 checked
        #print(len(self._images['MPRAGEPre']))

    def _check_im_shapes(self):
        shapes = [im.shape for im in self._images.values()]
        assert len(set(shapes)) == 1

    def __getitem__(self, index):   # slice idx
        data = [self._get_slices(imt, index) for imt in self.im_types]  # 3/subj
        if self.transform:
            data = self.transform(*data)
        return data

    def _get_slices(self, im_type, index):
        im_slice = self._images[im_type][index] #156，156，76  317=5/76
        name = '_'.join([im_slice.name, im_type])
        return NamedData(name=name, data=im_slice.data)

    def __len__(self):
        key = list(self._images.keys())[0]
        return len(self._images[key])


class Dataset(_Dataset):
    def __init__(
        self, subjects, datadict_indics, center_slice_keys, data_dtypes,
        apply_mask=False, shuffle_once=False, border=0, mask_rand=False,    #mask_rand？
        num=500,
    ):
        self.subjects = subjects    # -1,1 checked
        #print(subjects)
        self.datadict_indices = datadict_indics     #{'input': [0, 1], 'pred_truth': [2]}
        self.center_slice_keys = center_slice_keys
        self.data_dtypes = data_dtypes
        self.apply_mask = apply_mask
        self.shuffle_once = shuffle_once
        self.border = border
        self.mask_rand = mask_rand
        self.num = num  #None?
        num_slices = [len(sub) for sub in self.subjects]    # 388
        self._num_slices_cumsum = np.cumsum(num_slices) # 388 all
        if self.shuffle_once:
            self._ind_mapping = list(range(len(self)))  #2716=388*7
            random.shuffle(self._ind_mapping)

    def __len__(self):
        if self.mask_rand:
            return self.num
        else:
            return self._num_slices_cumsum[-1]  # 394=158+158+78acuum=16154

    def __getitem__(self, index):
        if self.shuffle_once:
            index = self._ind_mapping[index]
        sub_ind, slice_ind = self._split_index(index)
        #print(sub_ind)
        #print(slice_ind)
        slices = list(self.subjects[sub_ind][slice_ind])
        #print(len(slices ))
        mask = None
        if 'mask' in self.datadict_indices:
            # a = self.datadict_indices['mask'][0]
            #print(len(slices))
            mask = slices[self.datadict_indices['mask'][0]]

        if self.mask_rand:
            mask_ind = tuple(s//2 for s in mask.data.shape[1:])
            while mask.data[(0,) + mask_ind] < 0.5:
                slice_ind = random.randrange(len(self.subjects[sub_ind]))
                
                slices = list(self.subjects[sub_ind][slice_ind])
                mask = slices[self.datadict_indices['mask'][0]]

        data = dict()
        for k, v in self.datadict_indices.items():
            sub_slices = [slices[i] for i in v]     #k='input', v=[0,1]
            center_slice = k in self.center_slice_keys
            dtype = getattr(np, self.data_dtypes.get(k, 'float32'))
            border = self.border if 'truth' in k or 'mask' in k else 0
            data[k] = self._concat_data(sub_slices, mask, center_slice, dtype, border)

        if self.subjects[sub_ind].params is not None:
            data['params'] = self.subjects[sub_ind].params

        return data

    def _concat_data(self, data, mask=None, center_slice=False, dtype=None, border=0):
        names = [d.name.split('_') for d in data]
        prefix = '_'.join(names[0][:3])
        suffix = '_'.join(names[0][4:]) #''
        modalities = '-'.join([n[3] for n in names])
        name = '_'.join([prefix, modalities, suffix])
        if mask is not None and self.apply_mask:    # apply mre mask not brainmask
            data = [d.data * mask.data for d in data]
        if center_slice:    #for the slice in mid
            slice_ind = data[0].data.shape[0] // 2
            data = [d.data[slice_ind : slice_ind + 1] for d in data]
        else:
            data = [d.data for d in data]  # without nornalize
        data = np.concatenate(data, axis=0)
        data[data<0]=0
        assert data.min()>=0
        if border > 0:
            indexing = (slice(border, -border),) * (data.ndim - 1)
            data = data[(...,) + indexing]
        if dtype is not None:
            data = data.astype(dtype)
        return NamedData(name=name, data=data)

    def _split_index(self, index):
        sub_ind = np.searchsorted(self._num_slices_cumsum, index, side='right') #subject
        offset = 0 if sub_ind == 0 else self._num_slices_cumsum[sub_ind - 1]    #last subj idx
        slice_ind = index - offset
        return sub_ind, slice_ind

    def __str__(self):
        result = [f'{k}_ind={v}' for k, v in self.datadict_indices.items()]
        result = ',\n  '.join(result)
        result = (
            f'{self.__class__.__name__}(\n'
            f'  {result},\n'
            f'  center_slice_keys={self.center_slice_keys},\n'
            f'  shuffle_once={self.shuffle_once},\n'
            f'  len={len(self)}\n)'
        )
        return result
