import torch
import numpy as np
from model import BraTSCNN

def predict_tumor(model, image_path, device):
    """Predict segmentation for new brain scan"""
    
    # Load image
    img = np.load(image_path)
    img_tensor = torch.from_numpy(img).float().permute(3,0,1,2).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).cpu().numpy()[0]
    
    return prediction

# Use it
model = BraTSCNN(3, 4)
model.load_state_dict(torch.load('best_brats_cnn.pth', map_location='cpu'))

pred = predict_tumor(model, 'D:/ml-research-papers/brats_cnn/BraTS2020_TrainingData/test_data_3channels/images/image_0.npy', 'cpu')
print(f"Prediction shape: {pred.shape}")  # (128, 128, 128)
