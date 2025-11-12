import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURATION ---
# IMPORTANT: Update this path to your BRATS dataset folder!
# If you are using Google Drive, the path should look like: "/content/gdrive/MyDrive/BRATS_Data/"
DATA_PATH = "brats_cnn/MICCAI_BraTS2020_TrainingData"
TARGET_SIZE = (64, 64)
RANDOM_STATE = 42
TEST_SIZE = 0.20
BATCH_SIZE = 32

# --- HELPER FUNCTIONS ---

def normalize_slice(slice_data: np.ndarray) -> np.ndarray:
    """Performs Z-score normalization on a 2D slice."""
    mean = np.mean(slice_data)
    std = np.std(slice_data)
    # Check if std is zero to prevent division by zero (e.g., in blank slices)
    if std == 0:
        return slice_data
    # Apply Z-score: (x - mean) / std. Add epsilon to ensure stability.
    return (slice_data - mean) / (std + 1e-8)

def load_and_preprocess_data(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads, processes, and stacks 3 modalities into N x 64 x 64 x 3 NumPy arrays.
    """
    all_images = []
    all_labels = []

    print(f"Starting data processing from: {data_dir}")
    patient_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found {len(patient_folders)} patient folders.")

    for i, patient_folder in enumerate(patient_folders):
        patient_dir = os.path.join(data_dir, patient_folder)

        try:
            # --- 1. Load Raw 3D Volumes ---
            # Using T1, T1ce, and FLAIR for the 3 channels
            modalities = {
                'T1': nib.load(os.path.join(patient_dir, patient_folder + '_t1.nii')).get_fdata(),
                'T1ce': nib.load(os.path.join(patient_dir, patient_folder + '_t1ce.nii')).get_fdata(),
                'FLAIR': nib.load(os.path.join(patient_dir, patient_folder + '_flair.nii')).get_fdata(),
                'SEG': nib.load(os.path.join(patient_dir, patient_folder + '_seg.nii')).get_fdata()
            }

            num_slices = modalities['T1'].shape[2]

            # --- 2. Slice Extraction, Normalization, Resizing, and Stacking ---
            for z in range(num_slices):
                slice_T1 = modalities['T1'][:, :, z]
                slice_T1ce = modalities['T1ce'][:, :, z]
                slice_FLAIR = modalities['FLAIR'][:, :, z]
                slice_SEG = modalities['SEG'][:, :, z]

                # Filtering: Check if the slice contains any signal (i.e., is part of the brain)
                if np.max(slice_T1ce) > 0:

                    # Normalization (Task A.3)
                    norm_T1 = normalize_slice(slice_T1)
                    norm_T1ce = normalize_slice(slice_T1ce)
                    norm_FLAIR = normalize_slice(slice_FLAIR)

                    # Resize (Task A.4)
                    resized_T1 = resize(norm_T1, TARGET_SIZE, anti_aliasing=True, preserve_range=True)
                    resized_T1ce = resize(norm_T1ce, TARGET_SIZE, anti_aliasing=True, preserve_range=True)
                    resized_FLAIR = resize(norm_FLAIR, TARGET_SIZE, anti_aliasing=True, preserve_range=True)

                    # Stack Channels (Task A.4) -> Channels Last (H, W, C)
                    three_channel_tensor = np.stack(
                        [resized_T1, resized_T1ce, resized_FLAIR],
                        axis=-1
                    ).astype(np.float32)

                    # Labeling (Task A.6): 1 if tumor is present, 0 otherwise
                    label = 1 if np.max(slice_SEG) > 0 else 0

                    all_images.append(three_channel_tensor)
                    all_labels.append(label)

        except Exception as e:
            print(f"Error processing patient {patient_folder}: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(patient_folders)} patients. Current samples: {len(all_images)}")

    print("-" * 50)
    print(f"Total number of image slices extracted: {len(all_images)}")

    # Convert lists to final NumPy arrays
    X = np.array(all_images, dtype=np.float32)
    Y = np.array(all_labels, dtype=np.int32)

    print(f"Final Input Data (X) Shape: {X.shape}")
    print(f"Final Label Data (Y) Shape: {Y.shape}")
    return X, Y

# --- PYTORCH DATASET WRAPPER (FOR PHASE 2) ---

class MRIDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        # Convert NumPy arrays to PyTorch Tensors
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long() # Use long() for binary labels/targets

        # CRUCIAL PYTORCH STEP: Transpose Channels
        # PyTorch expects shape (N, C, H, W). NumPy output is (N, H, W, C).
        self.features = self.features.permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # unsqueeze(0) makes the label tensor shape (1,) for typical PyTorch loss functions
        return self.features[idx], self.labels[idx]


# --- MAIN EXECUTION ---

if __name__ == '__main__':
    # Load and preprocess all data
    X, Y = load_and_preprocess_data(DATA_PATH)

    # --- Train/Test Split (Task B) ---
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=Y  # Ensures label balance is preserved
    )

    print("-" * 50)
    print("Train/Test Split Complete.")
    print(f"Training Set Shape (X_train): {X_train.shape}")
    print(f"Testing Set Shape (X_test): {X_test.shape}")
    print("-" * 50)

    # --- PyTorch DataLoader Initialization ---
    train_dataset = MRIDataset(X_train, Y_train)
    test_dataset = MRIDataset(X_test, Y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count() or 1 # Use all available CPU cores for speed
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count() or 1
    )

    print("\nPyTorch DataLoaders Ready for Phase 2: Model Training.")
    print(f"Training Batches: {len(train_loader)}")
    print(f"Testing Batches: {len(test_loader)}")
    # Verify the final PyTorch tensor shape (Batch Size, Channels, Height, Width)
    print(f"Final PyTorch Input Shape: {next(iter(train_loader))[0].shape}")