import os
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy import stats


pop_name='U01_all'
exp='T1_T2'

template_pop_path="/home/alshareef/U01_Data/Population_average/ants_template/"+pop_name+"/01/proc/register_MRE/template_average_MV/"+pop_name+"_MRE_50Hz_stiff.nii"
template_pop_file=nib.load(template_pop_path).get_fdata()

outfn="/home/junyi/projects/MRE/Data/Register_MRE_"+pop_name+"/"+exp+"_distance"

pop=template_pop_file>0

if os.path.exists(outfn):
    os.remove(outfn)

subjects={'08','10','12','26','32','33','52','61'}
# subjects={'08','52'}    # mid_male
# subjects={'10','12','26','32','33','52'}
psnr_pop_ls=[]
ssim_pop_ls=[]
psnr_subj_ls=[]
ssim_subj_ls=[]
df=list()

for subj in subjects:

    template_path="/home/junyi/projects/MRE/Data/Register_MRE_"+pop_name+"/U01_UDEL_00"+subj+"_MRE_to_template_50Hz_stiff.nii.gz"
    template_file=nib.load(template_path).get_fdata()

    mre=template_file>0

    subject_path="/home/junyi/projects/MRE/Data/Register_MRE_"+pop_name+"/U01_UDEL_00"+subj+"_MRE_"+exp+"_to_template_50Hz_stiff.nii.gz"
    subject_file=nib.load(subject_path).get_fdata()

    # psnr_pop = peak_signal_noise_ratio(template_pop_file, subject_file, data_range=(template_pop_file.max()-template_pop_file.min()))
    # ssim_pop = structural_similarity(template_pop_file, subject_file, data_range=(template_pop_file.max()-template_pop_file.min()))

    # psnr_subj = peak_signal_noise_ratio(template_file, subject_file, data_range=(template_file.max()-template_file.min()))
    # ssim_subj = structural_similarity(template_file, subject_file, data_range=(template_file.max()-template_file.min()))

    psnr_pop = peak_signal_noise_ratio(template_pop_file[pop>0], subject_file[pop>0], data_range=(template_pop_file[pop>0].max()-template_pop_file[pop>0].min()))
    ssim_pop = structural_similarity(template_pop_file[pop>0], subject_file[pop>0], data_range=(template_pop_file[pop>0].max()-template_pop_file[pop>0].min()))

    psnr_subj = peak_signal_noise_ratio(template_file[mre>0], subject_file[mre>0], data_range=(template_file[mre>0].max()-template_file[mre>0].min()))
    ssim_subj = structural_similarity(template_file[mre>0], subject_file[mre>0], data_range=(template_file[mre>0].max()-template_file[mre>0].min()))

    psnr_pop_ls.append(psnr_pop)
    ssim_pop_ls.append(ssim_pop)
    psnr_subj_ls.append(psnr_subj)
    ssim_subj_ls.append(ssim_subj)

    df.append({'exp': exp, 'subj': subj, 'psnr_pop': psnr_pop, 'ssim_pop': ssim_pop,'psnr_subj': psnr_subj, 'ssim_subj': ssim_subj})
    print((exp, subj, psnr_pop, ssim_pop, psnr_subj, ssim_subj))

df = pd.DataFrame(df)
df = df.sort_values(['subj'])

avg_psnr_pop=sum(psnr_pop_ls)/len(psnr_pop_ls)
std_psnr_pop=np.std(psnr_pop_ls)
avg_ssim_pop=sum(ssim_pop_ls)/len(ssim_pop_ls)
std_ssim_pop=np.std(ssim_pop_ls)

avg_psnr_subj=sum(psnr_subj_ls)/len(psnr_subj_ls)
std_psnr_subj=np.std(psnr_subj_ls)
avg_ssim_subj=sum(ssim_subj_ls)/len(ssim_subj_ls)
std_ssim_subj=np.std(ssim_subj_ls)

df.loc[len(df.index)] = ['avg_std','psnr_pop',avg_psnr_pop,std_psnr_pop,'','']
print("avg_std", "psnr_pop", avg_psnr_pop, std_psnr_pop)

df.loc[len(df.index)] = ['avg_std','ssim_pop',avg_ssim_pop,std_ssim_pop,'','']
print("avg_std", "ssim_pop", avg_ssim_pop, std_ssim_pop)

df.loc[len(df.index)] = ['avg_std','psnr_subj',avg_psnr_subj,std_psnr_subj,'','']
print("avg_std", "psnr_subj", avg_psnr_subj, std_psnr_subj)

df.loc[len(df.index)] = ['avg_std','ssim_subj',avg_ssim_subj,std_ssim_subj,'','']
print("avg_std", "ssim_subj", avg_ssim_subj, std_ssim_subj)

df.to_csv(outfn, index=False)