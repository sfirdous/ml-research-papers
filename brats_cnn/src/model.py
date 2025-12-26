import torch
import torch.nn as nn
import torch.nn.functional as F

class BraTSCNN(nn.Module):
    """
    5-layer CNN for Brain Tumor Segmentation
    Based on ICASERT 2019 paper
    """
    
    def __init__(self,in_channels = 3,num_classes=4):
        super(BraTSCNN,self).__init__()
        
        # Layer 1: Convolution + ReLU  + MaxPool
        self.conv1 = nn.Conv3d(in_channels,32,kernel_size=3,padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=2,stride=2)
        
        # Layer 2: Convolution + ReLU  + MaxPool
        self.conv2 = nn.Conv3d(32,64,kernel_size = 3,padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=2,stride=2)
        
        # Layer 3: Convolution + ReLU  + MaxPool
        self.conv3 = nn.Conv3d(64,128,kernel_size=3,padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=2,stride=2)
        
        # Layer 4: Convolution + ReLU 
        self.conv4 = nn.Conv3d(128,256,kernel_size=3,padding=1)
        self.relu4 = nn.ReLU()
        
        # Layer 5: Convolution
        self.conv5 = nn.Conv3d(256,num_classes,kernel_size=1)
        
        # Upsampling to restore original size
        self.upsample = nn.Upsample(scale_factor=8,mode='trilinear',align_corners=False)
        
    def forward(self,x):
        # Layer 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Layer 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Layer 4
        x = self.conv4(x)
        x = self.relu4(x)
        
        # Layer 5
        x = self.conv5(x)
        
        # Upsample to original size
        x = self.upsample(x)
        
        return x
    
    