import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle
import pandas as pd

# global filePathMRI,filePathProc,filePathRaw
filePathProc = '/iacl/pg23/BrainBiomechanics/DATA_PROCESSED/MRE/UDEL'
filePathMRI='01/proc/register_to_MRE'
filePathRaw = '/iacl/pg23/BrainBiomechanics/DATA_RAW/MRE/UDEL'

class SubjData:
    def __init__(self, subject):
        self.subject = subject
        self.T1w = os.path.join(filePathProc,subject,filePathMRI,subject+"_01_MREreg_T1.nii.gz")
        self.T2w = os.path.join(filePathProc,subject,filePathMRI,subject+"_01_MREreg_T2.nii.gz")
        self.ADC = os.path.join(filePathProc,subject,filePathMRI,subject+"_01_MREreg_DT_AD.nii")
        self.FA = os.path.join(filePathProc,subject,filePathMRI,subject+"_01_MREreg_DT_FA.nii")
        # use MRE mask
        # self.brainmask = os.path.join(filePathProc,subject,filePathMRI,subject+"_01_MREreg_brainmask.nii.gz")
        self.stiff30 =os.path.join(filePathRaw,subject,"mre_spiral_AP_30Hz","mre_spiral_AP_30Hz_props_shear_stiff.nii.gz")
        self.stiff50 =os.path.join(filePathRaw,subject,"mre_spiral_AP_50Hz","mre_spiral_AP_50Hz_props_shear_stiff.nii.gz")
        self.stiff70 =os.path.join(filePathRaw,subject,"mre_spiral_AP_70Hz","mre_spiral_AP_70Hz_props_shear_stiff.nii.gz")
        self.isexist=self._check_exist()
        self.mremask=self._get_mremask()

    def _check_exist(self):    
        isexist=os.path.isfile(self.T1w) and os.path.isfile(self.T2w) \
                and os.path.isfile(self.ADC) and os.path.isfile(self.FA)\
                and os.path.isfile(self.stiff30) and os.path.isfile(self.stiff50) and os.path.isfile(self.stiff70) 
        return isexist
    
    def _get_mremask(self):    
        data = np.array(nib.load(self.stiff50).get_fdata().reshape(-1,1))
        return np.ma.make_mask(data)


data_T1w_all = []
data_T2w_all = []
data_ADC_all = []
data_FA_all = []
data_30Hz_all = []
data_50Hz_all = []
data_70Hz_all = []

data_T1pT2_all=[]
data_T1mT2_all=[]

savePath='/home/junyi/projects/MRE/Data/JointHist'
for root, dirs, files in os.walk('/iacl/pg23/BrainBiomechanics/DATA_RAW/MRE/UDEL'):
    for dir in dirs:
        if "U01_UDEL_00" in dir:
            data=SubjData(dir)
            # print(data.subject)
            if data.isexist:
                # get data
                data_T1w = np.array(np.array(nib.load(data.T1w).get_fdata().reshape(-1)))
                data_T2w = np.array(nib.load(data.T2w).get_fdata().reshape(-1))
                data_ADC = np.array(nib.load(data.ADC).get_fdata().reshape(-1))
                data_FA = np.array(nib.load(data.FA).get_fdata().reshape(-1))
                data_30Hz = np.array(nib.load(data.stiff30).get_fdata().reshape(-1))
                data_50Hz = np.array(nib.load(data.stiff50).get_fdata().reshape(-1))
                data_70Hz = np.array(nib.load(data.stiff70).get_fdata().reshape(-1))
                # get mask index
                data_mremask = np.array(data._get_mremask().reshape(-1))
                idx=np.where(data_mremask==True)
                data_T1w = data_T1w[idx]
                data_T2w = data_T2w[idx]
                data_ADC = data_ADC[idx]    
                data_FA = data_FA[idx]
                data_30Hz = data_30Hz[idx]
                data_50Hz = data_50Hz[idx]
                data_70Hz = data_70Hz[idx]
                data_ADC = data_ADC[data_ADC>0]
                data_FA = data_FA[data_FA>0]
                # # cal
                # data_T1pT2=data_T1w+data_T2w
                # data_T1mT2=data_T1w-data_T2w
                # # plot
                # fig = plt.figure(figsize=(12,3))
                # plt.title(str(data.subject))
                # # T1w-50
                # plt.subplot(1,6,1)
                # plt.hist2d(data_50Hz,data_T1w,bins=(800,800),cmap=plt.cm.Greys)
                # plt.xlabel("50Hz stiffness")
                # plt.ylabel("MPRAGE")
                # plt.xlim([0,5000])
                # plt.ylim([0,2000])
                # # T2w-50
                # plt.subplot(1,6,2)
                # plt.hist2d(data_50Hz,data_T2w,bins=(800,800),cmap=plt.cm.Greys)
                # plt.xlabel("50Hz stiffness")
                # plt.ylabel("T2w")
                # plt.xlim([0,5000])
                # plt.ylim([-300,8000])
                # # # T1+T2-50
                # # plt.subplot(2, 4, 2)
                # # plt.hist2d(data_50Hz_all,data_T1pT2_all,bins=(1000,1000),cmap=plt.cm.Greys)
                # # plt.xlabel("50Hz stiffness")
                # # plt.ylabel("T1w+T2w")
                # # plt.xlim([0,5000])
                # # plt.ylim([0,3500])
                # # # T1*T2-50
                # # plt.subplot(2, 4, 6)
                # # plt.hist2d(data_50Hz_all,data_T1mT2_all,bins=(5000,5000),cmap=plt.cm.Greys)
                # # plt.xlabel("50Hz stiffness")
                # # plt.ylabel("T1w*T2w")
                # # plt.xlim([0,5000])
                # # plt.ylim([0,1500000])
                # # ADC-50
                # plt.subplot(1,6,3)
                # plt.hist2d(data_50Hz,data_ADC,bins=(800,800),cmap=plt.cm.Greys)
                # plt.xlabel("50Hz stiffness")
                # plt.ylabel("ADC")
                # plt.xlim([0,5000])
                # plt.ylim([-100,2000])
                # # FA-50
                # plt.subplot(1,6,4)
                # plt.hist2d(data_50Hz,data_FA,bins=(100,100),cmap=plt.cm.Greys)
                # plt.xlabel("50Hz stiffness")
                # plt.ylabel("FA")
                # plt.xlim([0,5000])
                # plt.ylim([-0.05,1])
                # # 30-50
                # plt.subplot(1,6,5)
                # plt.hist2d(data_50Hz,data_30Hz,bins=(800,800),cmap=plt.cm.Greys)
                # plt.xlabel("50Hz stiffness")
                # plt.ylabel("30Hz stiffness")
                # plt.xlim([0,5000])
                # plt.ylim([0,3000])
                # # 70-50
                # plt.subplot(1,6,6)
                # plt.hist2d(data_50Hz,data_70Hz,bins=(800,800),cmap=plt.cm.Greys)
                # plt.xlabel("50Hz stiffness")
                # plt.ylabel("70Hz stiffness")
                # plt.xlim([0,5000])
                # plt.ylim([0,7000])
                # # show plot
                # plt.tight_layout()
                # plt.savefig(os.path.join(savePath,data.subject))

                # fig = plt.figure()
                # ax = fig.add_subplot(projection='3d')
                # # ax = Axes3D(fig)
                # surf=ax.scatter3D(data_T1w, data_T2w, data_50Hz,cmap=cm.jet)
                # fig.colorbar(surf)
                # ax.set_zlabel("50Hz stiffness")
                # ax.set_xlabel("T1w")
                # ax.set_ylabel("T2w")
                # ax.set_xlim(0,1500)
                # ax.set_ylim(0,2000)
                # ax.set_zlim(0,5000)
                # filename=data.subject+"_3d"
                # plt.tight_layout()
                # plt.savefig(os.path.join(savePath,filename))
                
                # df = np.column_stack((data_T1w,data_T2w,data_50Hz))
                # np.savetxt(os.path.join(savePath,'test.dat'), df, header = ' T1w, T2w, stiff50')
                
                data_T1w_all.append(data_T1w)
                data_T2w_all.append(data_T2w)
                data_ADC_all.append(data_ADC)
                data_FA_all.append(data_FA)
                data_30Hz_all.append(data_30Hz)
                data_50Hz_all.append(data_50Hz) 
                data_70Hz_all.append(data_70Hz)
                # data_T1pT2_all.append(data_T1pT2)
                # data_T1mT2_all.append(data_T1mT2)

    break

print("sum")

data_T1w_all = np.array(list(chain.from_iterable(data_T1w_all))).reshape(-1)
data_T2w_all = np.array(list(chain.from_iterable(data_T2w_all))).reshape(-1)
data_ADC_all = np.array(list(chain.from_iterable(data_ADC_all))).reshape(-1)
data_FA_all = np.array(list(chain.from_iterable(data_FA_all))).reshape(-1)
data_30Hz_all = np.array(list(chain.from_iterable(data_30Hz_all))).reshape(-1)
data_50Hz_all = np.array(list(chain.from_iterable(data_50Hz_all))).reshape(-1)
data_70Hz_all = np.array(list(chain.from_iterable(data_70Hz_all))).reshape(-1)
# data_T1pT2_all = np.array(list(chain.from_iterable(data_T1pT2_all))).reshape(-1)
# data_T1mT2_all = np.array(list(chain.from_iterable(data_T1mT2_all))).reshape(-1)


# df = np.column_stack((data_T1w_all,data_T2w_all,data_50Hz_all))
# np.savetxt(os.path.join(savePath,'all.dat'), df, header = ' T1w, T2w, stiff50')


# plot
fig = plt.figure(figsize=(12,2))
plt.title(str(data.subject))
# T1w-50
plt.subplot(1,6,1)
plt.hist2d(data_50Hz_all,data_T1w_all,bins=(800,800),cmap=plt.cm.Greys)
plt.xlabel("50Hz stiffness")
plt.ylabel("MPRAGE")
plt.xlim([0,5000])
plt.ylim([0,2000])
# T2w-50
plt.subplot(1,6,2)
plt.hist2d(data_50Hz_all,data_T2w_all,bins=(800,800),cmap=plt.cm.Greys)
plt.xlabel("50Hz stiffness")
plt.ylabel("T2w")
plt.xlim([0,5000])
plt.ylim([0,4000])
# # T1+T2-50
# plt.subplot(2, 4, 2)
# plt.hist2d(data_50Hz_all,data_T1pT2_all,bins=(1000,1000),cmap=plt.cm.Greys)
# plt.xlabel("50Hz stiffness")
# plt.ylabel("T1w+T2w")
# plt.xlim([0,5000])
# plt.ylim([0,3500])
# # T1*T2-50
# plt.subplot(2, 4, 6)
# plt.hist2d(data_50Hz_all,data_T1mT2_all,bins=(5000,5000),cmap=plt.cm.Greys)
# plt.xlabel("50Hz stiffness")
# plt.ylabel("T1w*T2w")
# plt.xlim([0,5000])
# plt.ylim([0,1500000])
# ADC-50
plt.subplot(1,6,3)
plt.hist2d(data_50Hz_all,data_ADC_all,bins=(800,800),cmap=plt.cm.Greys)
plt.xlabel("50Hz stiffness")
plt.ylabel("ADC")
plt.xlim([0,5000])
plt.ylim([0,1500])
# FA-50
plt.subplot(1,6,4)
plt.hist2d(data_50Hz_all,data_FA_all,bins=(100,100),cmap=plt.cm.Greys)
plt.xlabel("50Hz stiffness")
plt.ylabel("FA")
plt.xlim([0,5000])
plt.ylim([0,1])
# ADC-50
plt.subplot(1,6,5)
plt.hist2d(data_50Hz_all,data_30Hz_all,bins=(800,800),cmap=plt.cm.Greys)
plt.xlabel("50Hz stiffness")
plt.ylabel("30Hz stiffness")
plt.xlim([0,5000])
plt.ylim([0,3000])
# FA-50
plt.subplot(1,6,6)
plt.hist2d(data_50Hz_all,data_70Hz_all,bins=(800,800),cmap=plt.cm.Greys)
plt.xlabel("50Hz stiffness")
plt.ylabel("70Hz stiffness")
plt.xlim([0,5000])
plt.ylim([0,7000])
# show plot
plt.tight_layout()
plt.savefig(os.path.join(savePath,'jointhist_new2'))

# plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# step=int(data_T1w_all.shape[0]/800)
# x_data,y_data = np.meshgrid(np.arange(0,data_T1w_all.shape[0],step),
#                             np.arange(0,data_T2w_all.shape[0],step))
# for i in range(y_data.shape[0]):


# # z_data = data_50Hz_all.flatten()
# ax.bar3d( x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data )
# plt.show()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # ax = Axes3D(fig)
# ax.plot_trisurf(data_T1w_all, data_T2w_all, data_50Hz_all)
# ax.set_zlabel("50Hz stiffness")
# ax.set_xlabel("T1w")
# ax.set_ylabel("T2w")
# ax.set_xlim(0,1500)
# ax.set_ylim(0,2000)
# ax.set_zlim(0,5000)
# plt.savefig(os.path.join(savePath,'3d'))