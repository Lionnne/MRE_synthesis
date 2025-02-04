import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

data_path="/home/junyi/projects/MRE/Data/JointHist_syn"
# key_list=[]
# for root, dirs, files in os.walk(data_path):
#     for fn in files:
#         if "U01UDEL" in fn:
#             key=fn.split("_")[0]
#             if os.path.join(data_path,key) not in key_list:
#                 key_list.append(os.path.join(data_path,key))
title_dict=["Ventricles", "CSF", "Cerebellum Gray", "Cerebrum Cortical Gray",
            "Cerebrum Deep Gray", "Brainstem", "Cerebellum White", "Cerebrum White"]

key_list=['/home/junyi/projects/MRE/Data/JointHist_syn/U01UDEL0032']
for key in key_list:
    data_T1_syn=nib.load(key+"_T1_to_stiff_median.nii.gz").get_fdata()
    data_T1T2_syn=nib.load(key+"_T1_T2_to_stiff_median.nii.gz").get_fdata()
    data_T1T2ADFA_syn=nib.load(key+"_T1_T2_AD_FA_to_stiff_median.nii.gz").get_fdata()
    data_T130_syn=nib.load(key+"_T1_30Hz_stiff_to_stiff_median.nii.gz").get_fdata()

    data_gt=nib.load(key.replace("JointHist_syn","All_data")+"_50Hz.nii.gz").get_fdata()
    data_slant=nib.load(os.path.join(key.replace("JointHist_syn","All_data"),"01/proc",key.split("/")[-1]+"_T1_slant.nii.gz")).get_fdata()
    labels=np.unique(data_slant.astype("int"))
    label_key={}
    label_key[1]=np.array([4,11,49,50,51,52]) # Ventricles
    label_key[2]=() # CSF
    label_key[3]=np.array([38,39])    # Cerebellum Gray
    label_key[4]=labels[labels>=100]    # Cerebrum Cortical Gray
    label_key[5]=np.array([23,30,31,32,36,37,47,48,55,56,57,58,59,60,61,62,75,76])    # Cerebrum Deep Gray
    label_key[6]=np.array([35])   # Brainstem
    label_key[7]=np.array([40,41,71,72,73])  # Cerebellum white
    label_key[8]=np.array([44,45])    # Cerebrum white

    seg_x,seg_y={},{}
    cnt=0
    # fig = plt.figure(figsize=(54,36))
    # fig = plt.figure(figsize=(18,12))
    fig, ax = plt.subplots(2,3,figsize=(10,5.4))
    for k in label_key.items():
        if k[0]==1 or k[0]==2:
            continue
        data_x=()
        data_y=()
        for l in k[1]:
            if l==k[1][0]:
                data_x=data_gt[data_slant==l]
                data_y=data_T130_syn[data_slant==l]
                data_x2=data_x[data_x>0]
                data_y2=data_y[data_x>0]
            else:
                data_x=np.hstack([data_x,data_gt[data_slant==l]])
                data_y=np.hstack([data_y,data_T130_syn[data_slant==l]])
                data_x2=data_x[data_x>0]
                data_y2=data_y[data_x>0]
        # apply MRE-mask!
        seg_x[k[0]]=data_x2
        seg_y[k[0]]=data_y2
        # plt.subplot(2,3,k[0]-2,aspect='equal')
        axk = ax[int(k[0]/3)-1,k[0]%3].hist2d(seg_x[k[0]],seg_y[k[0]],bins=50,cmap='turbo', range=np.array([(0, 5000), (0, 5000)]))
        ax[int(k[0]/3)-1,k[0]%3].set_aspect('equal')
        # plt.hist2d(seg_x[k[0]],seg_y[k[0]],bins=100,cmap='turbo', range=np.array([(0, 5000), (0, 5000)]))
        # plt.plot((0,5000),(0,5000),'r-')
        ax[int(k[0]/3)-1,k[0]%3].set(xlim=(0, 5000), ylim=(0, 5000))
        ax[int(k[0]/3)-1,k[0]%3].set_title(title_dict[k[0]-1])
        cbar=fig.colorbar(axk[3], ax=ax[int(k[0]/3)-1,k[0]%3])
        density = cbar.get_ticks()
        axk[3].set_clim(density.min(), density.max())
        # ax[int(k[0]/3)-1,k[0]%3].hist2d(seg_x[k[0]],seg_y[k[0]],bins=50,cmap='turbo', range=np.array([(0, 5000), (0, 5000)]),vmin=density.min(),vmax=density.max())
    # for idx in range(len(labels)):
    #     seg_x[labels[idx]]=data_gt[data_slant==labels[idx]]
    #     seg_y[labels[idx]]=data_T130_syn[data_slant==labels[idx]]
    #     if labels[idx]!=0:
    #             cnt+=1
    #             plt.subplot(12,11,cnt)
    #             # plt.scatter(seg_x[labels[idx]],seg_y[labels[idx]],marker=".",alpha=0.5)
    #             plt.hist2d(seg_x[labels[idx]],seg_y[labels[idx]],bins=100, cmap='turbo',range=np.array([(0, 5000), (0, 5000)]))
    #             plt.plot((0,5000),(0,5000),'r-')
    #             plt.xlim(0,5000)
    #             plt.ylim(0,5000)
    #             plt.title("label "+str(labels[idx]))
    # plt.subplots_adjust(top=0.98,bottom=0.015,left=0.02,right=0.96)
    # cax=plt.axes((0.972, 0.015, 0.015, 0.965))
    # plt.colorbar(cax=cax, orientation='vertical')
    # plt.suptitle(key.split("/")[-1]+" joint histogram X:GT Y:T130syn",y=0.995)
    # plt.subplots_adjust(top=0.95,bottom=0.05,left=0.04,right=0.93)
    # cax=plt.axes((0.95, 0.05, 0.015, 0.9))
    # plt.colorbar(cax=cax,orientation='vertical')
    plt.suptitle(key.split("/")[-1]+" joint histogram X:GT Y:T130_syn",y=0.988)
    plt.tight_layout()
    plt.savefig(os.path.join("/home/junyi/projects/MRE/Data/JointHist_syn",key.split("/")[-1]+"_xGT_yT130syn_hist6"))