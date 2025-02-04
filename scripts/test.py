#!/usr/bin/env python3

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filenames', nargs='+')
parser.add_argument('-d', '--indir')
parser.add_argument('-p', '--params', nargs='+')
parser.add_argument('-c', '--config')
parser.add_argument('-C', '--checkpoint')
parser.add_argument('-o', '--outdir')
parser.add_argument('-O', '--outnames', nargs='+')
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()


import nibabel as nib
import numpy as np
from pathlib import Path

from tofgatir.trainers import EvaluatorBuilder

from tofgatir.utils import map_data


Path(args.outdir).mkdir(parents=True, exist_ok=True)
if args.filenames:
    builder = EvaluatorBuilder(args)
    evaluator, model = builder.build()
    result = evaluator.run()
    # print(result)
    ref_obj = nib.load(args.filenames[0])

    for k, v in result.items():
        counter = 0
        for subk, subv in v.items():
            for data in subv:
                # data=map_data(data)
                outname = args.outnames[counter]
                outname = '_'.join([outname, k])
                outname = Path(args.outdir, outname).with_suffix('.nii.gz')
                out_obj = nib.Nifti1Image(np.abs(data), ref_obj.affine, ref_obj.header)
                print(outname)
                out_obj.to_filename(outname)
                counter += 1
else:
    modalities=args.config.split("/")[-1].split("_")
    modalities=modalities[1:-2]
    modalities=[m+"_stiff" if "Hz" in m else m for m in modalities]
    for root, dirs,files in os.walk(args.indir):
        subjects=[f.split("_")[0] for f in files]
        subjects=set(subjects)
        for subj in subjects:
            suffix=[m+".nii" if "A" in m else m+".nii.gz" for m in modalities]
            filenames=[os.path.join(args.indir,subj+"_"+s) for s in suffix]
            args.filenames=filenames
            ref_obj = nib.load(args.filenames[0])
            print("\n")
            # inference
            builder = EvaluatorBuilder(args)
            evaluator, model = builder.build()
            result = evaluator.run()
            # print(result)
            print("\n")
            for k, v in result.items():
                counter = 0
                outsuffix = '_'.join(m for m in modalities)+"_to_stiff"
                for subk, subv in v.items():
                    for data in subv:
                        # data=data.astype(np.float32)
                        outname = '_'.join([subj,outsuffix, k])
                        outname = Path(args.outdir, outname).with_suffix('.nii.gz')
                        out_obj = nib.Nifti1Image(data, ref_obj.affine, ref_obj.header)
                        # out_obj.header.set_data_dtype(np.float32)
                        print(outname)
                        out_obj.to_filename(outname)
                        counter += 1

        break