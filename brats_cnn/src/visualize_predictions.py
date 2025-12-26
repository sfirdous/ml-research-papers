
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from model import BraTSCNN

def visualize_prediction(model, image_path, mask_path, device):
    """Visualize model prediction vs ground truth"""
    
    # Load data
    img = np.load(image_path)
    mask = np.load(mask_path)
    
    # Prepare for model
    img_tensor = torch.from_numpy(img).float().permute(3, 0, 1, 2).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).cpu().numpy()[0]
    
    # Ground truth
    ground_truth = np.argmax(mask, axis=-1)
    
    # Visualize middle slice
    slice_idx = 64
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Input modalities
    axes[0, 0].imshow(img[:, :, slice_idx, 0], cmap='gray')
    axes[0, 0].set_title('FLAIR')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img[:, :, slice_idx, 1], cmap='gray')
    axes[0, 1].set_title('T1CE')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img[:, :, slice_idx, 2], cmap='gray')
    axes[0, 2].set_title('T2')
    axes[0, 2].axis('off')
    
    # Row 2: Segmentation
    axes[1, 0].imshow(ground_truth[:, :, slice_idx], cmap='tab10', vmin=0, vmax=3)
    axes[1, 0].set_title('Ground Truth')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(prediction[:, :, slice_idx], cmap='tab10', vmin=0, vmax=3)
    axes[1, 1].set_title('Prediction')
    axes[1, 1].axis('off')
    
    # Difference map
    diff = (ground_truth != prediction).astype(float)
    axes[1, 2].imshow(diff[:, :, slice_idx], cmap='Reds')
    axes[1, 2].set_title('Errors (Red)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png', dpi=150)
    print("✅ Saved visualization to 'prediction_visualization.png'")
    plt.show()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = BraTSCNN(in_channels=3, num_classes=4).to(device)
    model.load_state_dict(torch.load('best_brats_cnn.pth', map_location=device))
    
    # Test on one sample
    image_path = Path('BraTS2020_TrainingData/input_data_3channels/images/image_0.npy')
    mask_path = Path('BraTS2020_TrainingData/input_data_3channels/masks/mask_0.npy')
    
    visualize_prediction(model, image_path, mask_path, device)
