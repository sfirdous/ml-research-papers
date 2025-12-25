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

os.makedirs('BraTS2020_TrainingData/input_data_3channels/images', exist_ok=True)
os.makedirs('BraTS2020_TrainingData/input_data_3channels/masks', exist_ok=True)


t2_list = sorted(glob.glob('MICCAI_BraTS2020_TrainingData/*/*t2.nii'))
# print(len(t2_list))
t1ce_list = sorted(glob.glob('MICCAI_BraTS2020_TrainingData/*/*t1ce.nii'))
# print(len(t1ce_list))
flair_list = sorted(glob.glob('MICCAI_BraTS2020_TrainingData/*/*flair.nii'))
# print(len(flair_list))
mask_list = sorted(glob.glob('MICCAI_BraTS2020_TrainingData/*/*seg.nii'))
# print(len(mask_list))

for img in range(len(t2_list)):
    print('Now preparing img and mask number: ',img)
    
    temp_image_t2 = nib.load(t2_list[img]).get_fdata()
    temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1,temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
    
    temp_image_flair = nib.load(flair_list[img]).get_fdata()
    temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1,temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
    
    temp_image_t1ce = nib.load(t1ce_list[img]).get_fdata()
    temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1,temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
    
    temp_mask = nib.load(mask_list[img]).get_fdata()
    temp_mask=temp_mask.astype(np.uint8)
    
    temp_mask[temp_mask==4] = 3    # Reassign mask values 4 to 3
    
    temp_combined_x = np.stack([temp_image_flair,temp_image_t1ce,temp_image_t2],axis =3) 
    
    temp_combined_x = temp_combined_x[56:184,56:184,13:141]

    temp_mask = temp_mask[56:184,56:184,13:141]
    
    val,counts = np.unique(temp_mask,return_counts = True)
    
    if(1-(counts[0]/counts.sum())) > 0.01:
        print("Save me")
        temp_mask = F.one_hot(torch.from_numpy(temp_mask).long(),num_classes=4)
        temp_mask = temp_mask.numpy()

        np.save(f'BraTS2020_TrainingData/input_data_3channels/images/image_{img}.npy',temp_combined_x)
        np.save(f'BraTS2020_TrainingData/input_data_3channels/masks/mask_{img}.npy',temp_mask)
    else:
        print('I am useless')

        






