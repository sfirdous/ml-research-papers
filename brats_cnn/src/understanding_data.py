import numpy as np
import nibabel as nib
import glob 
import torch
import torch.nn.functional as F
import  matplotlib.pyplot as plt
from tifffile import imwrite

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

TRAIN_DATASET_PATH = 'D:/ml-research-papers/brats_cnn/MICCAI_BraTS2020_TrainingData/'

test_image_flair = nib.load(TRAIN_DATASET_PATH +'BraTS20_Training_001/BraTS20_Training_001_flair.nii').get_fdata()
# print(test_image_flair.max())

test_image_flair = scaler.fit_transform(test_image_flair.reshape(-1,test_image_flair.shape[-1])).reshape(test_image_flair.shape)
# print(test_image_flair.max())

test_image_t1 = nib.load(TRAIN_DATASET_PATH +'BraTS20_Training_001/BraTS20_Training_001_t1.nii').get_fdata()
test_image_t1 = scaler.fit_transform(test_image_t1.reshape(-1,test_image_t1.shape[-1])).reshape(test_image_t1.shape)

test_image_t1ce = nib.load(TRAIN_DATASET_PATH +'BraTS20_Training_001/BraTS20_Training_001_t1ce.nii').get_fdata()
test_image_t1ce = scaler.fit_transform(test_image_t1ce.reshape(-1,test_image_t1ce.shape[-1])).reshape(test_image_t1ce.shape)

test_image_t2 = nib.load(TRAIN_DATASET_PATH +'BraTS20_Training_001/BraTS20_Training_001_t2.nii').get_fdata()
test_image_t2 = scaler.fit_transform(test_image_t2.reshape(-1,test_image_t2.shape[-1])).reshape(test_image_t2.shape)

test_mask = nib.load(TRAIN_DATASET_PATH +'BraTS20_Training_001/BraTS20_Training_001_seg.nii').get_fdata()
test_mask=test_mask.astype(np.uint8)

# print(np.unique(test_mask))
test_mask[test_mask==4] = 3    # Reassign mask values 4 to 3
# print(np.unique(test_mask))

import random
n_slice = random.randint(0,test_mask.shape[2]-1)

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(test_image_flair[:,:,n_slice],cmap='gray')
plt.title('Image flair')

plt.subplot(2,3,2)
plt.imshow(test_image_t1[:,:,n_slice],cmap='gray')
plt.title('Image t1')

plt.subplot(2,3,3)
plt.imshow(test_image_t1ce[:,:,n_slice],cmap='gray')
plt.title('Image t1ce')

plt.subplot(2,3,4)
plt.imshow(test_image_t2[:,:,n_slice],cmap='gray')
plt.title('Image t2')

plt.subplot(2,3,5)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Image Mask')

plt.show()

# part 2

combined_x = np.stack([test_image_flair,test_image_t1ce,test_image_t2],axis =3) 
# print(combined_x.shape)

combined_x = combined_x[56:184,56:184,13:141]
# print(combined_x.shape)
test_mask = test_mask[56:184,56:184,13:141]

plt.figure(figsize=(12,8))

plt.subplot(221)
plt.imshow(combined_x[:,:,n_slice,0],cmap='gray')
plt.title('Image flair')

plt.subplot(222)
plt.imshow(combined_x[:,:,n_slice,1],cmap='gray')
plt.title('Image t1ce')

plt.subplot(223)
plt.imshow(combined_x[:,:,n_slice,2],cmap='gray')
plt.title('Image t2')

plt.subplot(224)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')

plt.show()

imwrite('BraTS2020_TrainingData/combined001.tif',combined_x)
np.save('BraTS2020_TrainingData/combined001.npy',combined_x)

my_img = np.load('BraTS2020_TrainingData/combined001.npy')
# print(my_img.shape)

test_mask = F.one_hot(torch.from_numpy(test_mask).long(),num_classes=4)