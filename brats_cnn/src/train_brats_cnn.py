import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
import time
from model import BraTSCNN


class BraTSDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_files = sorted(Path(image_dir).glob('*.npy'))
        self.mask_files = sorted(Path(mask_dir).glob('*.npy'))
        assert len(self.image_files) == len(self.mask_files)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img = np.load(self.image_files[idx])
        mask = np.load(self.mask_files[idx])
        img = torch.from_numpy(img).float().permute(3, 0, 1, 2)
        mask = torch.from_numpy(mask).float().permute(3, 0, 1, 2)
        return img, mask


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = torch.argmax(masks.to(device), dim=1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'  Batch [{batch_idx}/{len(dataloader)}] - Loss: {loss.item():.4f}')
    
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks_indices = torch.argmax(masks.to(device), dim=1)
            
            outputs = model(images)
            loss = criterion(outputs, masks_indices)
            running_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == masks_indices).sum().item()
            total += masks_indices.numel()
    
    accuracy = 100.0 * correct / total
    return running_loss / len(dataloader), accuracy


def main():
    print("="*60)
    print("BraTS CNN Training - Direct Split (No File Copying)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✅ Using device: {device}")
    
    # Load dataset
    print(f"\n📂 Loading dataset...")
    dataset = BraTSDataset(
        image_dir='BraTS2020_TrainingData/input_data_3channels/images',
        mask_dir='BraTS2020_TrainingData/input_data_3channels/masks'
    )
    
    print(f"   Total samples: {len(dataset)}")
    
    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"   Train: {train_size} samples")
    print(f"   Val:   {val_size} samples")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    # Model
    print(f"\n🏗️ Building model...")
    model = BraTSCNN(in_channels=3, num_classes=4).to(device)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3,factor=0.5)
    
    # Training loop
    print(f"\n🚀 Starting training...\n")
    best_val_loss = float('inf')
    
    NUM_EPOCHS = 20
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print("-" * 60)
        
        start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        print(f"\n  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Time: {time.time()-start:.1f}s\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_brats_cnn.pth')
            print(f"  ✅ Saved best model\n")
    
    print("="*60)
    print(f"✅ Training complete! Best val loss: {best_val_loss:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
