#!/usr/bin/env python3

import nibabel as nib
import pandas as pd
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy import stats
import numpy as np
import os
import torch

def cal_iou(mask1,mask2):
    intersection=mask1*mask2
    union=mask1+mask2
    intersection_area=np.sum(intersection)
    union_area=np.sum(union)
    iou=intersection_area/union_area
    return iou

preddir = 'Results/T1_to_50Hz_mre_0320'
datadir = 'Data/test_all'

outfn =preddir.split('/')[0]+"/test_metrics_median_"+preddir.split('/')[1]

if os.path.exists(outfn):
    os.remove(outfn)

df = list()
ssimls=[]
psnrls=[]
ssimlsmre=[]
psnrlsmre=[]
ssimlsinter=[]
psnrlsinter=[]
ssimlsrmv=[]
psnrlsrmv=[]
ssimlsinterrmv=[]
psnrlsinterrmv=[]
iouls=[]

for subjdir in Path(preddir).iterdir():
    if "median" in subjdir.name and "61" not in subjdir.name:
        pred = nib.load(subjdir).get_fdata()    # prediction
        pred = np.nan_to_num(pred)
        truth_path = f'{datadir}/{subjdir.name}'
        truth_path = truth_path.replace('T1_to_stiff_median','50Hz_stiff')
        truth_path = Path(truth_path)
        truth = nib.load(truth_path).get_fdata()

        slant_path = 'Data/All_data/'+subjdir.name.split('_')[0]+'/01/proc/'+subjdir.name.split('_')[0]+'_T1_slant.nii.gz'
        slant_path=Path(slant_path)
        slant = nib.load(slant_path).get_fdata()
        slant_rmv = (slant==4)+(slant==11)+(slant==49)+(slant==50)+(slant==51)+(slant==52)
        pred_rmv=pred*(1-slant_rmv.astype(int))
        truth_rmv=truth*(1-slant_rmv.astype(int))
        
        mask=truth>0
        mask_syn=pred>0
        mask_inter=mask*mask_syn

        psnr = peak_signal_noise_ratio(truth, pred, data_range=(truth.max()-truth.min()))
        ssim = structural_similarity(truth, pred, data_range=(truth.max()-truth.min()))

        mrepsnr = peak_signal_noise_ratio(truth[mask>0], pred[mask>0], data_range=(truth[mask>0].max()-truth[mask>0].min()))
        mressim = structural_similarity(truth[mask>0], pred[mask>0], data_range=(truth[mask>0].max()-truth[mask>0].min()))

        interpsnr = peak_signal_noise_ratio(truth[mask_inter>0], pred[mask_inter>0], data_range=(truth[mask_inter>0].max()-truth[mask_inter>0].min()))
        interssim = structural_similarity(truth[mask_inter>0], pred[mask_inter>0], data_range=(truth[mask_inter>0].max()-truth[mask_inter>0].min()))

        rmvpsnr = peak_signal_noise_ratio(truth_rmv[mask>0], pred_rmv[mask>0], data_range=(truth_rmv[mask>0].max()-truth_rmv[mask>0].min()))
        rmvssim = structural_similarity(truth_rmv[mask>0], pred_rmv[mask>0], data_range=(truth_rmv[mask>0].max()-truth_rmv[mask>0].min()))

        rmvinterpsnr = peak_signal_noise_ratio(truth_rmv[mask_inter>0], pred_rmv[mask_inter>0], data_range=(truth_rmv[mask_inter>0].max()-truth_rmv[mask_inter>0].min()))
        rmvinterssim = structural_similarity(truth_rmv[mask_inter>0], pred_rmv[mask_inter>0], data_range=(truth_rmv[mask_inter>0].max()-truth_rmv[mask_inter>0].min()))

        psnrls.append(psnr)
        ssimls.append(ssim)
        psnrlsmre.append(mrepsnr)
        ssimlsmre.append(mressim)
        psnrlsinter.append(interpsnr)
        ssimlsinter.append(interssim)
        psnrlsrmv.append(rmvpsnr)
        ssimlsrmv.append(rmvssim)
        psnrlsinterrmv.append(rmvinterpsnr)
        ssimlsinterrmv.append(rmvinterssim)

        iou=cal_iou(mask,mask_syn)
        iouls.append(iou)

        df.append({'exp': (preddir.split('/')[1]).split('_l')[0], 'subj': subjdir.name.split('_')[0], 'psnr': psnr, 'ssim': ssim,'mre-masked psnr': mrepsnr, 'mre-masked ssim': mressim,'intersection psnr': interpsnr, 'intersection ssim': interssim,'mre-masked w/o ventricle psnr': rmvpsnr, 'mre-masked w/o ventricle ssim': rmvssim,'intersection w/o ventricle psnr': rmvinterpsnr, 'intersection w/o ventricle ssim': rmvinterssim,'mask iou':iou})
        print((preddir.split('/')[1]).split('-')[0], subjdir.name.split('_')[0], psnr, ssim, mrepsnr, mressim, interpsnr, interssim, rmvpsnr, rmvssim, rmvinterpsnr,rmvinterssim, iou)

df = pd.DataFrame(df)
df = df.sort_values(['subj'])

avg_ssim=sum(ssimls)/len(ssimls)
avg_psnr=sum(psnrls)/len(psnrls)
std_ssim=np.std(ssimls)
std_psnr=np.std(psnrls)

df.loc[len(df.index)] = ['','average',avg_psnr,avg_ssim,'','','','','','','','','']
print("average", avg_psnr, avg_ssim)

df.loc[len(df.index)] = ['','standard deviation',std_psnr,std_ssim,'','','','','','','','','']
print("standard deviation", std_psnr, std_ssim)

avg_mressim=sum(ssimlsmre)/len(ssimlsmre)
avg_mrepsnr=sum(psnrlsmre)/len(psnrlsmre)
std_mressim=np.std(ssimlsmre)
std_mrepsnr=np.std(psnrlsmre)

df.loc[len(df.index)] = ['','mre-masked average',avg_mrepsnr,avg_mressim,'','','','','','','','','']
print("mre-masked average", avg_mrepsnr, avg_mressim)

df.loc[len(df.index)] = ['','mre-masked standard deviation',std_mrepsnr,std_mressim,'','','','','','','','','']
print("mre-masked standard deviation", std_mrepsnr, std_mressim)

avg_interssim=sum(ssimlsinter)/len(ssimlsinter)
avg_interpsnr=sum(psnrlsinter)/len(psnrlsinter)
std_interssim=np.std(ssimlsinter)
std_interpsnr=np.std(psnrlsinter)

df.loc[len(df.index)] = ['','intersection average',avg_interpsnr,avg_interssim,'','','','','','','','','']
print("intersection average", avg_interpsnr, avg_interssim)

df.loc[len(df.index)] = ['','intersection standard deviation',std_interpsnr,std_interssim,'','','','','','','','','']
print("intersection standard deviation", std_interpsnr, std_interssim)

avg_rmvssim=sum(ssimlsrmv)/len(ssimlsrmv)
avg_rmvpsnr=sum(psnrlsrmv)/len(psnrlsrmv)
std_rmvssim=np.std(ssimlsrmv)
std_rmvpsnr=np.std(psnrlsrmv)

df.loc[len(df.index)] = ['','mre-masked w/o ventricle average',avg_rmvpsnr,avg_rmvssim,'','','','','','','','','']
print("mre-masked w/o ventricle average", avg_rmvpsnr, avg_rmvssim)

df.loc[len(df.index)] = ['','mre-masked w/o ventricle standard deviation',std_rmvpsnr,std_rmvssim,'','','','','','','','','']
print("mre-masked w/o ventricle standard deviation", std_rmvpsnr, std_rmvssim)

avg_interrmvssim=sum(ssimlsinterrmv)/len(ssimlsinterrmv)
avg_interrmvpsnr=sum(psnrlsinterrmv)/len(psnrlsinterrmv)
std_interrmvssim=np.std(ssimlsinterrmv)
std_interrmvpsnr=np.std(psnrlsinterrmv)

df.loc[len(df.index)] = ['','intersection w/o ventricle average',avg_interrmvpsnr,avg_interrmvssim,'','','','','','','','','']
print("intersection w/o ventricle average", avg_interrmvpsnr, avg_interrmvssim)

df.loc[len(df.index)] = ['','intersection w/o ventricle standard deviation',std_interrmvpsnr,std_interrmvssim,'','','','','','','','','']
print("intersection w/o ventricle standard deviation", std_interrmvpsnr, std_interrmvssim)

avg_iou=sum(iouls)/len(iouls)
std_iou=np.std(iouls)

df.loc[len(df.index)] = ['','mask iou average',avg_iou,'','','','','','','','','','']
print("mask iou average", avg_iou)

df.loc[len(df.index)] = ['','mask iou standard deviation',std_iou,'','','','','','','','','','']
print("mask iou standard deviation", std_iou)

df.to_csv(outfn, index=False)
