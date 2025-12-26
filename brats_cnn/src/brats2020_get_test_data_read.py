import numpy as np
import nibabel as nib
import glob 
import torch
import torch.nn.functional as F
import  matplotlib.pyplot as plt
from tifffile import imwrite
import os 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

os.makedirs('BraTS2020_TrainingData/test_data_3channels/images', exist_ok=True)
os.makedirs('BraTS2020_TrainingData/test_data_3channels/masks', exist_ok=True)


t2_list = sorted(glob.glob('MICCAI_BraTS2020_ValidationData/*/*t2.nii'))
# print(len(t2_list))
t1ce_list = sorted(glob.glob('MICCAI_BraTS2020_ValidationData/*/*t1ce.nii'))
# print(len(t1ce_list))
flair_list = sorted(glob.glob('MICCAI_BraTS2020_ValidationData/*/*flair.nii'))
# print(len(flair_list))

for img in range(len(t2_list)):
    print('Now preparing img number: ',img)
    
    temp_image_t2 = nib.load(t2_list[img]).get_fdata()
    temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1,temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
    
    temp_image_flair = nib.load(flair_list[img]).get_fdata()
    temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1,temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
    
    temp_image_t1ce = nib.load(t1ce_list[img]).get_fdata()
    temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1,temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
    

    
    temp_combined_x = np.stack([temp_image_flair,temp_image_t1ce,temp_image_t2],axis =3) 
    
    temp_combined_x = temp_combined_x[56:184,56:184,13:141]
    

    np.save(f'BraTS2020_TrainingData/test_data_3channels/images/image_{img}.npy',temp_combined_x)
    

        






