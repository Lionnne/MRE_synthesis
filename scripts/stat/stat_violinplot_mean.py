import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pandas as pd

# data_path="/home/junyi/projects/MRE/Data/test_T1_T2_AD_FA_50Hz"

### 1.10
T1syn_path="/home/junyi/projects/MRE/Data/JointHist_syn/T1syn_mre/metric_T1syn_mre_mean"
T1T2syn_path="/home/junyi/projects/MRE/Data/JointHist_syn/T1T2syn_mre/metric_T1T2syn_mre_mean"
T1T2ADFAsyn_path="/home/junyi/projects/MRE/Data/JointHist_syn/T1T2ADFAsyn_mre/metric_T1T2ADFAsyn_mre_mean"
T130syn_path="/home/junyi/projects/MRE/Data/JointHist_syn/T130syn_mre/metric_T130syn_mre_mean"
# all_path="/home/junyi/projects/MRE/Data/All_data"

# title_dict=["Ventricles", "CSF", "Cerebellum Gray", "Cerebrum Cortical Gray",
            # "Cerebrum Deep Gray", "Brainstem", "Cerebellum White", "Cerebrum White"]

T1syn_data=pd.read_csv(T1syn_path)
T1syn=T1syn_data.iloc[1:22:2]
T1T2syn_data=pd.read_csv(T1T2syn_path)
T1T2syn=T1T2syn_data[1:22:2]
T1T2ADFAsyn_data=pd.read_csv(T1T2ADFAsyn_path)
T1T2ADFAsyn=T1T2ADFAsyn_data[1:22:2]
T130syn_data=pd.read_csv(T130syn_path)
T130syn=T130syn_data[1:22:2]

GT=T1syn_data[0:21:2]

# key_list=[]
# for root, dirs, files in os.walk(data_path):
#     for fn in files:
#         if "U01UDEL" in fn:
#             key=fn.split("_")[0]
#             if key not in key_list:
#                 key_list.append(key)

# for key in key_list:
#     # print(key+":")
#     data_T1syn=nib.load(T1syn_path+"/"+key+"_T1_to_stiff_median.nii.gz").get_fdata()
#     data_T1T2syn=nib.load(T1T2syn_path+"/"+key+"_T1_T2_to_stiff_median.nii.gz").get_fdata()
#     data_T1T2ADFAsyn=nib.load(T1T2ADFAsyn_path+"/"+key+"_T1_T2_AD_FA_to_stiff_median.nii.gz").get_fdata()
#     data_T130syn=nib.load(T130syn_path+"/"+key+"_T1_30Hz_stiff_to_stiff_median.nii.gz").get_fdata()
    
#     data_gt=nib.load(all_path+"/"+key+"_50Hz.nii.gz").get_fdata()
#     mask=(data_gt>0).astype('int')*(data_T1syn>0).astype('int')*(data_T1T2syn>0).astype('int')*(data_T1T2ADFAsyn>0).astype('int')*(data_T130syn>0).astype('int')
#     # data_syn=(data_xxxsyn)*mask
#     data_syn=data_T1T2ADFAsyn*mask
    
#     data_slant=nib.load(os.path.join(all_path,key,"01/proc",key.split("/")[-1]+"_T1_slant.nii.gz")).get_fdata()
#     labels=np.unique(data_slant.astype("int"))
#     label_key={}
#     label_key[1]=np.array([4,11,49,50,51,52]) # Ventricles
#     label_key[2]=() # CSF
#     label_key[3]=np.array([38,39])    # Cerebellum Gray
#     label_key[4]=labels[labels>=100]    # Cerebrum Cortical Gray
#     label_key[5]=np.array([23,30,31,32,36,37,47,48,55,56,57,58,59,60,61,62,75,76])    # Cerebrum Deep Gray
#     label_key[6]=np.array([35])   # Brainstem
#     label_key[7]=np.array([40,41,71,72,73])  # Cerebellum white
#     label_key[8]=np.array([44,45])    # Cerebrum white

#     seg_y,seg_f,seg_f_fitted={},{},{}
#     # fig, ax = plt.subplots(2,3,figsize=(9.5,5))
#     for k in label_key.items():
#         if k[0]==1 or k[0]==2:
#             continue
#         data_x=()
#         data_y=()
#         for l in k[1]:
#             if l==k[1][0]:
#                 data_x=data_gt[data_slant==l]   # (n,)
#                 data_y=data_syn[data_slant==l]  # (n,) same number
#                 data_x2=data_x[data_x>0]    # slant & truth mask
#                 data_y2=data_y[data_x>0]    # synth apply slant & truth mask
#                 data_y3=data_y2[data_y2>0]  # slant & truth & synth mask
#                 data_x3=data_x2[data_y2>0]  # truth apply slant & truth & synth mask
#             else:
#                 data_x=np.hstack([data_x,data_gt[data_slant==l]])
#                 data_y=np.hstack([data_y,data_syn[data_slant==l]])
#                 data_x2=data_x[data_x>0]
#                 data_y2=data_y[data_x>0]
#                 data_y3=data_y2[data_y2>0]
#                 data_x3=data_x2[data_y2>0]
#         # apply MRE-mask!
#         seg_y[k[0]]=data_x3     # all ground truth
#         seg_f[k[0]]=data_y3     # all synthesized

df=list()
for i in range(0,21,2):   
    df.append({'': 'Ground Truth', 'region': 'Cerebellum Gray', 'stiffness': GT['Cerebellum Gray'][i]})
    df.append({'': 'Ground Truth', 'region': 'Cerebrum Cortical Gray', 'stiffness': GT['Cerebrum Cortical Gray'][i]})
    df.append({'': 'Ground Truth', 'region': 'Cerebrum Deep Gray', 'stiffness': GT['Cerebrum Deep Gray'][i]})
    df.append({'': 'Ground Truth', 'region': 'Brainstem', 'stiffness': GT['Brainstem'][i]})
    df.append({'': 'Ground Truth', 'region': 'Cerebellum White', 'stiffness': GT['Cerebellum White'][i]})
    df.append({'': 'Ground Truth', 'region': 'Cerebrum White', 'stiffness': GT['Cerebrum White'][i]})

    df.append({'': 'Synthesized', 'region': 'Cerebellum Gray', 'stiffness': T130syn['Cerebellum Gray'][i+1]})
    df.append({'': 'Synthesized', 'region': 'Cerebrum Cortical Gray', 'stiffness': T130syn['Cerebrum Cortical Gray'][i+1]})
    df.append({'': 'Synthesized', 'region': 'Cerebrum Deep Gray', 'stiffness': T130syn['Cerebrum Deep Gray'][i+1]})
    df.append({'': 'Synthesized', 'region': 'Brainstem', 'stiffness': T130syn['Brainstem'][i+1]})
    df.append({'': 'Synthesized', 'region': 'Cerebellum White', 'stiffness': T130syn['Cerebellum White'][i+1]})
    df.append({'': 'Synthesized', 'region': 'Cerebrum White', 'stiffness': T130syn['Cerebrum White'][i+1]})
data=pd.DataFrame(df)

plt.figure(figsize=(12.8,6.5))
sns.violinplot(data=data, x='region', y='stiffness', hue='')
# plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11],labels=label)
# plt.title("Cerebellum Gray   Cerebrum Cortical Gray  Cerebrum Deep Gray         Brainstem           Cerebellum White        Cerebrum White",x=0.5,y=-0.13)
# plt.ylabel("Stiffness")
# plt.ylim(0,6000)
plt.grid(visible=True)
out_path ="/home/junyi/projects/MRE/Data/JointHist_syn/T130syn_mre"
plt.savefig(os.path.join(out_path,"xGT_yT130syn_mre_3mask_violinplot_mean"))
# plt.suptitle(key.split("/")[-1]+" joint histogram X:GT Y:T1T2ADFAsyn_mre",y=0.98)
# plt.tight_layout()
# plt.savefig(os.path.join(out_path,key.split("/")[-1]+"_xGT_yT1T2syn_mre_hist6_3mask"))

