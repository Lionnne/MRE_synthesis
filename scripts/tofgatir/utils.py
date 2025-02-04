import numpy as np
import yaml
from collections import namedtuple
from easydict import EasyDict
from collections.abc import Iterable
import matplotlib.pyplot as plt
import torch
from copy import deepcopy


LOGGER_NAME = 'tofgatir'
NamedData = namedtuple('NamedData', ['name', 'data'])


def padcrop(image, target_shape, use_channels=True):
    shape = image.shape[-2:] if use_channels else image.shape
    bbox = calc_padcrop_bbox(shape, target_shape)
    lpads = [max(0, 0 - b.start) for b in bbox]
    rpads = [max(0, b.stop - s) for s, b in zip(shape, bbox)]
    pad_width = tuple(zip(lpads, rpads))
    c_bbox = tuple(slice(b.start + l, b.stop + l) for b, l in zip(bbox, lpads))
    if use_channels:
        pad_width = ((0, 0), ) * (image.ndim - 2) + pad_width
        c_bbox = (..., ) + c_bbox
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=image.min())    
    # padded_image = np.pad(image, pad_width, mode='edge')
    cropped_image = padded_image[c_bbox]
    return cropped_image


def calc_padcrop_bbox(source_shape, target_shape):
    bbox = list()
    for ss, ts in zip(source_shape, target_shape):
        diff = ss - ts
        left = np.abs(diff) // 2 * np.sign(diff)
        right = left + ts
        bbox.append(slice(left, right))
    return tuple(bbox)


def dump_easydict_to_yml(filename, ed):
    def to_dict(value):
        if isinstance(value, EasyDict):
            result = dict()
            for k, v in value.items():
                result[k] = to_dict(v)
        elif isinstance(value, Iterable) and not isinstance(value, str):
            result = [to_dict(v) for v in value]
        else:
            result = value
        return result

    with open(filename, 'w') as f:
        yaml.dump(to_dict(ed), f)


def move_data_dict_to_cuda(data):
    for k, v in data.items():
        if isinstance(v, dict):
            data[k] = move_data_dict_to_cuda(v)
        else:
            data[k] = v.cuda()
    return data


def move_nameddata_dict_to_cuda(data):
    for k, v in data.items():
        if isinstance(v, dict):
            data[k] = move_data_dict_to_cuda(v)
        else:
            data[k] = NamedData(name=v.name, data=v.data.cuda())
    return data


def map_data(data):
    shape=data.shape
    data=[d/np.max(d) if np.max(d)>0 else d for d in data]
    data=np.reshape(data,shape)
    return  data

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

def cal_std_mse(data1,data2):   #tensor cuda
    mse=list()
    for i in range(data1.shape[0]):
        mse.append(np.square(np.subtract(data1[i],data2[i])).mean())
    return np.std(mse)

def cal_std_ce(data1,data2):   #tensor cuda
    ce=list()
    for i in range(data1.shape[0]):
        p = softmax(data1[i])
        ce.append(- np.mean(np.log(p) * data2[i]))
    return np.std(ce)
