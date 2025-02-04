import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

data_path="/home/junyi/projects/MRE/Data/test_T1_30Hz_50Hz"
# syn_path="/home/junyi/projects/MRE/Results/T1_to_50HZstiff-f300"
# syn_path="/home/junyi/projects/MRE/Results/T1_T2_to_50HZstiff-deep-l2-200"
# syn_path="/home/junyi/projects/MRE/Results/T1_T2_AD_FA_to_50HZstiff-10200557"
# syn_path="/home/junyi/projects/MRE/Results/T1_30Hz_to_50HZstiff-deep-l2-120"

### 1.10
T1syn_path="/home/junyi/projects/MRE/Results/T1_to_50HZstiff_mre-400"
T1T2syn_path="/home/junyi/projects/MRE/Results/T1_T2_to_50HZstiff_mre-60"
T1T2ADFAsyn_path="/home/junyi/projects/MRE/Results/T1_T2_AD_FA_to_50HZstiff_mre-30"
T130syn_path="/home/junyi/projects/MRE/Results/T1_30Hz_to_50HZstiff_mre-25"

all_path="/home/junyi/projects/MRE/Data/All_data"
title_dict=["Ventricles", "CSF", "Cerebellum Gray", "Cerebrum Cortical Gray",
            "Cerebrum Deep Gray", "Brainstem", "Cerebellum White", "Cerebrum White"]

out_path ="/home/junyi/projects/MRE/Data/JointHist_syn/T1syn_mre"

key_list=[]
for root, dirs, files in os.walk(data_path):
    for fn in files:
        if "U01UDEL" in fn and "0061" not in fn:
            key=fn.split("_")[0]
            if key not in key_list:
                key_list.append(key)

r_squared_all=np.zeros((len(key_list),6))
truth_mean_all=np.zeros((len(key_list),6))
pred_mean_all=np.zeros((len(key_list),6))
cnt=0
df=list()
df2=list()
for key in key_list:
    # print(key+":")
    data_T1syn=nib.load(T1syn_path+"/"+key+"_T1_to_stiff_median.nii.gz").get_fdata()
    data_T1T2syn=nib.load(T1T2syn_path+"/"+key+"_T1_T2_to_stiff_median.nii.gz").get_fdata()
    data_T1T2ADFAsyn=nib.load(T1T2ADFAsyn_path+"/"+key+"_T1_T2_AD_FA_to_stiff_median.nii.gz").get_fdata()
    data_T130syn=nib.load(T130syn_path+"/"+key+"_T1_30Hz_stiff_to_stiff_median.nii.gz").get_fdata()

    data_gt=nib.load(all_path+"/"+key+"_50Hz.nii.gz").get_fdata()
    mask=(data_gt>0).astype('int')*(data_T1syn>0).astype('int')*(data_T1T2syn>0).astype('int')*(data_T1T2ADFAsyn>0).astype('int')
    # data_syn=(data_xxxsyn)*mask
    data_syn=data_T1T2ADFAsyn*mask

    data_slant=nib.load(os.path.join(all_path,key,"01/proc",key.split("/")[-1]+"_T1_slant.nii.gz")).get_fdata()
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

    seg_y,seg_f,seg_f_fitted={},{},{}
    fig, ax = plt.subplots(2,3,figsize=(9.5,5))
    for k in label_key.items():
        if k[0]==1 or k[0]==2:
            continue
        data_x=()
        data_y=()
        for l in k[1]:
            if l==k[1][0]:
                data_x=data_gt[data_slant==l]   # (n,)
                data_y=data_syn[data_slant==l]  # (n,) same number
                data_x2=data_x[data_x>0]    # slant & truth mask
                data_y2=data_y[data_x>0]    # synth apply slant & truth mask
                data_y3=data_y2[data_y2>0]  # slant & truth & synth mask
                data_x3=data_x2[data_y2>0]  # truth apply slant & truth & synth mask
            else:
                data_x=np.hstack([data_x,data_gt[data_slant==l]])
                data_y=np.hstack([data_y,data_syn[data_slant==l]])
                data_x2=data_x[data_x>0]
                data_y2=data_y[data_x>0]
                data_y3=data_y2[data_y2>0]
                data_x3=data_x2[data_y2>0]
        # apply MRE-mask!
        seg_y[k[0]]=data_x3
        seg_f[k[0]]=data_y3
        # calculate
        model = LinearRegression()
        model.fit(seg_y[k[0]].reshape(-1,1), seg_f[k[0]].reshape(-1,1)) # y=wx+b
        w=model.coef_[0][0]
        b=model.intercept_[0]
        seg_f_fitted[k[0]]=w*seg_y[k[0]]+b
        ax[int(k[0]/3)-1,k[0]%3].plot(seg_y[k[0]],seg_f_fitted[k[0]],'r-')
        ax[int(k[0]/3)-1,k[0]%3].plot(seg_y[k[0]],seg_y[k[0]],'m--')
        r_squared=model.score(seg_y[k[0]].reshape(-1,1), seg_f[k[0]].reshape(-1,1))
        r_squared_all[cnt,k[0]-3]=r_squared
        # plot
        # axk = ax[int(k[0]/3)-1,k[0]%3].hist2d(seg_y[k[0]],seg_f[k[0]],bins=50,cmap='turbo', range=np.array([(0, 5000), (0, 5000)]))
        # ax[int(k[0]/3)-1,k[0]%3].set_aspect('equal')
        # ax[int(k[0]/3)-1,k[0]%3].set(xlim=(0, 5000), ylim=(0, 5000))
        # ax[int(k[0]/3)-1,k[0]%3].set_title(title_dict[k[0]-1]+", R2="+format(r_squared,'.2f'))
        # cbar=fig.colorbar(axk[3], ax=ax[int(k[0]/3)-1,k[0]%3])
        # density = cbar.get_ticks()
        # axk[3].set_clim(density.min(), density.max())

        # mean over label
        truth_mean=np.mean(seg_y[k[0]])
        pred_mean=np.mean(seg_f[k[0]])
        truth_mean_all[cnt,k[0]-3]=truth_mean
        pred_mean_all[cnt,k[0]-3]=pred_mean
        # print(title_dict[k[0]-1]+": truth mean="+str(truth_mean)+", prediction mean="+str(pred_mean))

    # plt.suptitle(key.split("/")[-1]+" joint histogram X:GT Y:T1syn_mre",y=0.98)
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_path,key.split("/")[-1]+"_xGT_yT1syn_mre_hist6_3mask"))

    df.append({'subj': key,
           'Cerebellum Gray': r_squared_all[cnt,0], 'Cerebrum Cortical Gray': r_squared_all[cnt,1],
           'Cerebrum Deep Gray': r_squared_all[cnt,2], 'Brainstem': r_squared_all[cnt,3],
           'Cerebellum White': r_squared_all[cnt,4], 'Cerebrum White': r_squared_all[cnt,5]})

    df2.append({'subj': key,
           'Cerebellum Gray': truth_mean_all[cnt,0], 'Cerebrum Cortical Gray': truth_mean_all[cnt,1],
           'Cerebrum Deep Gray': truth_mean_all[cnt,2], 'Brainstem': truth_mean_all[cnt,3],
           'Cerebellum White': truth_mean_all[cnt,4], 'Cerebrum White': truth_mean_all[cnt,5]})
    df2.append({'subj': key,
           'Cerebellum Gray': pred_mean_all[cnt,0], 'Cerebrum Cortical Gray': pred_mean_all[cnt,1],
           'Cerebrum Deep Gray': pred_mean_all[cnt,2], 'Brainstem': pred_mean_all[cnt,3],
           'Cerebellum White': pred_mean_all[cnt,4], 'Cerebrum White': pred_mean_all[cnt,5]})

    cnt+=1

r_squared_mean=np.mean(r_squared_all,axis=0)
r_squared_std=np.std(r_squared_all,axis=0)
df = pd.DataFrame(df)
df = df.sort_values(['subj'])
df.loc[len(df.index)] = ['mean',r_squared_mean[0],r_squared_mean[1],r_squared_mean[2],
                         r_squared_mean[3],r_squared_mean[4],r_squared_mean[5]]
df.loc[len(df.index)] = ['standard deviation',r_squared_std[0],r_squared_std[1],r_squared_std[2],
                         r_squared_std[3],r_squared_std[4],r_squared_std[5]]
outfn=os.path.join(out_path,"metric_T1T2ADFAsyn_mre_3mask")
if os.path.exists(outfn):
    os.remove(outfn)
df.to_csv(outfn, index=False)

truth_mean_mean=np.mean(truth_mean_all,axis=0)
pred_mean_mean=np.mean(pred_mean_all,axis=0)
df2 = pd.DataFrame(df2)
df2 = df2.sort_values(['subj'])
df2.loc[len(df2.index)] = ['truth mean',truth_mean_mean[0],truth_mean_mean[1],truth_mean_mean[2],
                         truth_mean_mean[3],truth_mean_mean[4],truth_mean_mean[5]]
df2.loc[len(df2.index)] = ['pred mean',pred_mean_mean[0],pred_mean_mean[1],pred_mean_mean[2],
                         pred_mean_mean[3],pred_mean_mean[4],pred_mean_mean[5]]

diff_mean=abs(truth_mean_mean-pred_mean_mean)
df2.loc[len(df2.index)] = ['mean difference abs',diff_mean[0],diff_mean[1],diff_mean[2],
                         diff_mean[3],diff_mean[4],diff_mean[5]]
diff_mean=abs(truth_mean_mean-pred_mean_mean)/truth_mean_mean
df2.loc[len(df2.index)] = ['mean difference perc',diff_mean[0],diff_mean[1],diff_mean[2],
                         diff_mean[3],diff_mean[4],diff_mean[5]]

outfn=os.path.join(out_path,"metric_T1T2ADFAsyn_mre_mean")
if os.path.exists(outfn):
    os.remove(outfn)
df2.to_csv(outfn, index=False)