from torch.utils.data import Dataset,DataLoader
import numpy as np
from pathlib import Path
import torch

class BraTSDataset(Dataset):
    def __init__(self,image_dir,mask_dir):
        self.image_files = sorted(Path(image_dir).glob('*.npy'))
        self.mask_files = sorted(Path(mask_dir).glob('*.npy'))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self,idx):
        img = np.load(self.image_files[idx])
        mask = np.load(self.mask_files[idx])
        
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        
        return img,mask

train_dataset = BraTSDataset('BraTS2020_TrainingData/input_data_3channels/images','BraTS2020_TrainingData/input_data_3channels/masks')
train_loader = DataLoader(train_dataset,batch_size=8,shuffle=True)

# print(len(train_dataset))
# print(len(train_dataset.image_files))
# print(len(train_dataset.mask_files))

# img,mask = train_dataset[0]
# print(type(img))
# print(img.dtype)
# print(img.shape)

# print(type(mask))
# print(mask.dtype)
# print(mask.shape)
# print(torch.unique(mask))

test_loader = DataLoader(train_dataset,batch_size=4,shuffle=False)

# images_batch,masks_batch = next(iter(test_loader))
# print(f"Batch images shape: {images_batch.shape}")
# print(f"Batch masks shape: {masks_batch.shape}")
# print(f"Batch images dtype: {images_batch.dtype}")

# batch_count = 0
# sample_count = 0

# for images,masks in test_loader:
#     batch_count += 1
#     sample_count += images.shape[0]

# print(f"Total batches: {batch_count}")
# print(f"Total samples seen: {sample_count}")
# print(f"Dataset length: {len(train_dataset)}")
# print(f"Match: {sample_count == len(train_dataset)}")
    
