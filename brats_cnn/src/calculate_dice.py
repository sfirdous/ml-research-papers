
import torch
import numpy as np
from pathlib import Path
from model import BraTSCNN

def dice_coefficient(pred, target, class_idx):
    """Calculate Dice score for one class"""
    pred_class = (pred == class_idx).astype(float)
    target_class = (target == class_idx).astype(float)
    
    intersection = (pred_class * target_class).sum()
    union = pred_class.sum() + target_class.sum()
    
    if union == 0:
        return 1.0  # No tumor in ground truth or prediction
    
    dice = 2 * intersection / union
    return dice

def evaluate_model(model, dataset_path, device):
    """Evaluate on all validation data"""
    
    image_dir = Path(dataset_path) / 'images'
    mask_dir = Path(dataset_path) / 'masks'
    
    image_files = sorted(image_dir.glob('*.npy'))
    mask_files = sorted(mask_dir.glob('*.npy'))
    
    model.eval()
    dice_scores = {0: [], 1: [], 2: [], 3: []}
    
    with torch.no_grad():
        for img_file, mask_file in zip(image_files[:20], mask_files[:20]):  # Test on 20 samples
            # Load
            img = np.load(img_file)
            mask = np.load(mask_file)
            
            # Predict
            img_tensor = torch.from_numpy(img).float().permute(3,0,1,2).unsqueeze(0).to(device)
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]
            
            # Ground truth
            gt = np.argmax(mask, axis=-1)
            
            # Dice per class
            for class_idx in range(4):
                dice = dice_coefficient(pred, gt, class_idx)
                dice_scores[class_idx].append(dice)
    
    # Print results
    print("\n" + "="*50)
    print("Dice Scores (Higher is better, max=1.0)")
    print("="*50)
    print(f"Class 0 (Background): {np.mean(dice_scores[0]):.4f}")
    print(f"Class 1 (Necrotic):   {np.mean(dice_scores[1]):.4f}")
    print(f"Class 2 (Edema):      {np.mean(dice_scores[2]):.4f}")
    print(f"Class 3 (Enhancing):  {np.mean(dice_scores[3]):.4f}")
    print(f"\nMean Dice Score:      {np.mean([np.mean(v) for v in dice_scores.values()]):.4f}")
    print("="*50)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BraTSCNN(in_channels=3, num_classes=4).to(device)
    model.load_state_dict(torch.load('best_brats_cnn.pth', map_location=device))
    
    evaluate_model(model, 'BraTS2020_TrainingData/input_data_3channels', device)
