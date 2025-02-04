#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-dir')
parser.add_argument('-o', '--output-dir')
args = parser.parse_args()

import re
import nibabel as nib
import numpy as np
from pathlib import Path
from copy import deepcopy

import os
os.mkdir(args.input_dir+"_norm")

dtype_str = 'float32'
# dtype = getattr(np, dtype_str)
Path(args.output_dir).mkdir(exist_ok=True, parents=True)
print(Path(args.input_dir).glob('*.nii*'))
for fn in Path(args.input_dir).glob('*.nii*'):
    print(fn)
    info = nib.load(fn)
    data = info.get_fdata()
    maxv=data.max()
    minv=data.min()
    if minv<0:
        data=data*(data>0)
    assert data.min()==0
    if "AD" not in str(fn).split("/")[-1]:
        data=data/maxv
        # data=data*2-1   # -1, 1 range
        # assert data.max()==1 and data.min()==-1
        assert data.max()==1 and data.min()==0

    # dtype_str = data.dtype.name
    outfn = re.sub(r'\.nii(\.gz)*$', '_data.dat', fn.name)
    outfn = Path(args.output_dir, outfn)
    fp = np.memmap(outfn, dtype=dtype_str, mode='w+', shape=data.shape)
    fp[:] = data[:]
    fp.flush()

    shape_outfn = re.sub(r'_data\.dat$', '_shape.npy', str(outfn))
    np.save(shape_outfn, data.shape)

    dtype_outfn = re.sub(r'_data\.dat$', '_dtype.txt', str(outfn))
    with open(dtype_outfn, 'w') as f:
        f.write(dtype_str)
    
    outname = str(fn).replace("/U01","_norm/U01")
    out_obj = nib.Nifti1Image(data, info.affine, info.header)
    out_obj.to_filename(outname)
