import torch
import numpy as np
import matplotlib.pyplot as plt
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


def visualize_prediction(img, gt, pred, dice_scores, sample_idx, save_dir):
    """Create and save visualization comparing ground truth and prediction"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Use middle slice for visualization
    slice_idx = img.shape[2] // 2
    img_slice = img[:, :, slice_idx, 0]  # First channel

    # Class names and colors
    class_names = ['Background', 'Necrotic', 'Edema', 'Enhancing']

    # Row 1: Input image and ground truth
    axes[0, 0].imshow(img_slice, cmap='gray')
    axes[0, 0].set_title('Input MRI Slice', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gt[:, :, slice_idx], cmap='jet', vmin=0, vmax=3)
    axes[0, 1].set_title('Ground Truth Segmentation', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(pred[:, :, slice_idx], cmap='jet', vmin=0, vmax=3)
    axes[0, 2].set_title('Predicted Segmentation', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2: Overlay comparisons
    axes[1, 0].imshow(img_slice, cmap='gray')
    axes[1, 0].imshow(gt[:, :, slice_idx], cmap='jet', alpha=0.4, vmin=0, vmax=3)
    axes[1, 0].set_title('Ground Truth Overlay', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(img_slice, cmap='gray')
    axes[1, 1].imshow(pred[:, :, slice_idx], cmap='jet', alpha=0.4, vmin=0, vmax=3)
    axes[1, 1].set_title('Prediction Overlay', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Dice scores text
    axes[1, 2].axis('off')
    dice_text = "Dice Scores:\n\n"
    for i, name in enumerate(class_names):
        dice_text += f"{name}: {dice_scores[i]:.4f}\n"
    dice_text += f"\nMean: {np.mean(list(dice_scores.values())):.4f}"

    axes[1, 2].text(0.5, 0.5, dice_text, 
                    fontsize=12, 
                    ha='center', 
                    va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Sample {sample_idx + 1} - Segmentation Results', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    # Save
    save_path = save_dir / f'prediction_sample_{sample_idx + 1:03d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def evaluate_model(model, dataset_path, device, num_samples=20, save_visualizations=True):
    """Evaluate on all validation data"""

    image_dir = Path(dataset_path) / 'images'
    mask_dir = Path(dataset_path) / 'masks'

    image_files = sorted(image_dir.glob('*.npy'))
    mask_files = sorted(mask_dir.glob('*.npy'))

    # Create output directory
    output_dir = Path('evaluation_results')
    output_dir.mkdir(exist_ok=True)

    model.eval()
    dice_scores = {0: [], 1: [], 2: [], 3: []}

    print(f"\nEvaluating {num_samples} samples...")
    print("="*60)

    with torch.no_grad():
        for idx, (img_file, mask_file) in enumerate(zip(image_files[:num_samples], mask_files[:num_samples])):
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
            sample_dice = {}
            for class_idx in range(4):
                dice = dice_coefficient(pred, gt, class_idx)
                dice_scores[class_idx].append(dice)
                sample_dice[class_idx] = dice

            # Visualize and save
            if save_visualizations:
                save_path = visualize_prediction(img, gt, pred, sample_dice, idx, output_dir)
                print(f"Sample {idx+1}/{num_samples} - Mean Dice: {np.mean(list(sample_dice.values())):.4f} - Saved: {save_path.name}")

    # Print summary results
    print("\n" + "="*60)
    print("FINAL DICE SCORES (Higher is better, max=1.0)")
    print("="*60)
    print(f"Class 0 (Background): {np.mean(dice_scores[0]):.4f} ± {np.std(dice_scores[0]):.4f}")
    print(f"Class 1 (Necrotic):   {np.mean(dice_scores[1]):.4f} ± {np.std(dice_scores[1]):.4f}")
    print(f"Class 2 (Edema):      {np.mean(dice_scores[2]):.4f} ± {np.std(dice_scores[2]):.4f}")
    print(f"Class 3 (Enhancing):  {np.mean(dice_scores[3]):.4f} ± {np.std(dice_scores[3]):.4f}")
    print(f"\nMean Dice Score:      {np.mean([np.mean(v) for v in dice_scores.values()]):.4f}")
    print("="*60)

    if save_visualizations:
        print(f"\n✅ All visualizations saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = BraTSCNN(in_channels=3, num_classes=4).to(device)
    model.load_state_dict(torch.load('best_brats_cnn.pth', map_location=device))

    # Evaluate and save visualizations
    evaluate_model(
        model, 
        'BraTS2020_TrainingData/input_data_3channels', 
        device,
        num_samples=20,  # Number of samples to evaluate
        save_visualizations=True  # Set to False to skip PNG saving
    )