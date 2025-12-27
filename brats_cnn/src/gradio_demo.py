import gradio as gr
import torch
import numpy as np
import cv2

from model import BraTSCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = BraTSCNN(in_channels=3, num_classes=4)
model.load_state_dict(torch.load("D:/ml-research-papers/brats_cnn/best_brats_cnn.pth", map_location=device))
model.to(device)
model.eval()

# Preprocess .npy
def preprocess_npy(npy_path):
    volume = np.load(npy_path).astype(np.float32)  # (H, W, D, C)
    print("Loaded .npy shape:", volume.shape)

    # normalize
    volume /= volume.max()  # scale 0-1 (robust)
    
    # tensor for model: [1, C, H, W, D]
    tensor = torch.from_numpy(volume).permute(3,0,1,2).unsqueeze(0).to(device)

    # middle slice (for visualization)
    mid_slice_idx = volume.shape[2] // 2
    # scale to 0-255 for visualization
    mid_slice = (volume[:, :, mid_slice_idx, 0] * 255).astype(np.uint8)
    return tensor, mid_slice

# Overlay utility
def create_overlay(image, mask):
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    colors = {
        1: (255, 0, 0),    # Necrotic
        2: (0, 255, 0),    # Edema
        3: (255, 255, 0)   # Enhancing
    }

    overlay = image_color.copy()
    for label, color in colors.items():
        overlay[mask == label] = color

    return cv2.addWeighted(image_color, 0.6, overlay, 0.4, 0)

# Prediction
def predict_segmentation(npy_file):
    input_tensor, base_image = preprocess_npy(npy_file)

    with torch.no_grad():
        output = model(input_tensor)  # [1, C, H, W, D]
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]  # [H, W, D]

    mid_mask = pred[:, :, pred.shape[2] // 2].astype(np.uint8)

    # Enhance visibility by converting mask to colored overlay
    overlay = create_overlay(base_image, mid_mask)

    # Stretch mask for display (optional)
    mask_display = (mid_mask * 85).astype(np.uint8)  # 0→0,1→85,2→170,3→255

    return mask_display, overlay

# Gradio UI
demo = gr.Interface(
    fn=predict_segmentation,
    inputs=gr.File(label="Upload BraTS .npy file"),
    outputs=[
        gr.Image(label="Predicted Mask"),
        gr.Image(label="Overlay Visualization")
    ],
    title="BraTS Brain Tumor Segmentation Demo",
    description="Upload a BraTS .npy volume. Shows middle slice segmentation."
)

demo.launch()
