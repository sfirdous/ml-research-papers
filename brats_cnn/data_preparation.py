import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
from sklearn.model_selection import train_test_split # Not used for splitting, but useful for imports
import torch
from torch.utils.data import Dataset
from glob import glob


# --- CONFIGURATION ---
DATA_PATH = "brats_cnn/MICCAI_BraTS2020_TrainingData"
OUTPUT_DATA_DIR = "brats_cnn/processed_brats_data" 

# New output file: saves the entire dataset without splitting
FULL_OUTPUT_FILE = 'brats_cnn/brats_full_data.npz'

TARGET_SIZE = (64, 64)
RANDOM_STATE = 42 # Retained for consistency, though not used for split
BATCH_SIZE = 32


# --- HELPER FUNCTIONS ---

def normalize_slice(slice_data: np.ndarray) -> np.ndarray:
    """Performs Z-score normalization on a 2D slice."""
    mean = np.mean(slice_data)
    std = np.std(slice_data)
    if std == 0:
        return slice_data
    return (slice_data - mean) / (std + 1e-8)

def load_and_preprocess_data(data_dir: str, output_dir: str) -> list[str]:
    """
    Loads, processes, and saves data patient-by-patient to avoid RAM crash.
    Returns a list of file paths to the saved data.
    """
    saved_file_paths = []
    print(f"Starting data processing from: {data_dir}")
    
    try:
        patient_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    except FileNotFoundError:
        print(f"ERROR: Data directory not found at {data_dir}. Check DATA_PATH.")
        return []

    print(f"Found {len(patient_folders)} patient folders.")

    for i, patient_folder in enumerate(patient_folders):
        patient_dir = os.path.join(data_dir, patient_folder)
        
        # Check if this patient file already exists
        file_path = os.path.join(output_dir, f"{patient_folder}_data.npz")
        if os.path.exists(file_path):
            saved_file_paths.append(file_path)
            continue # Skip processing if file is already saved

        # Lists to hold slices for the CURRENT patient only
        patient_images = []
        patient_labels = []

        try:
            # Load Raw 3D Volumes
            modalities = {
                'T1': nib.load(os.path.join(patient_dir, patient_folder + '_t1.nii')).get_fdata(),
                'T1ce': nib.load(os.path.join(patient_dir, patient_folder + '_t1ce.nii')).get_fdata(),
                'FLAIR': nib.load(os.path.join(patient_dir, patient_folder + '_flair.nii')).get_fdata(),
                'SEG': nib.load(os.path.join(patient_dir, patient_folder + '_seg.nii')).get_fdata()
            }

            num_slices = modalities['T1'].shape[2]

            # Slice Extraction, Preprocessing, and Stacking
            for z in range(num_slices):
                slice_T1ce = modalities['T1ce'][:, :, z] 

                # Filtering: Check if the slice contains any brain signal
                if np.max(slice_T1ce) > 0:
                    slice_T1 = modalities['T1'][:, :, z]
                    slice_FLAIR = modalities['FLAIR'][:, :, z]
                    slice_SEG = modalities['SEG'][:, :, z]
                    
                    # Normalization (Z-Score)
                    norm_T1 = normalize_slice(slice_T1)
                    norm_T1ce = normalize_slice(slice_T1ce)
                    norm_FLAIR = normalize_slice(slice_FLAIR)

                    # Resize & Stack (H, W, C)
                    resized_T1 = resize(norm_T1, TARGET_SIZE, anti_aliasing=True, preserve_range=True)
                    resized_T1ce = resize(norm_T1ce, TARGET_SIZE, anti_aliasing=True, preserve_range=True)
                    resized_FLAIR = resize(norm_FLAIR, TARGET_SIZE, anti_aliasing=True, preserve_range=True)

                    three_channel_tensor = np.stack(
                        [resized_T1, resized_T1ce, resized_FLAIR],
                        axis=-1
                    ).astype(np.float32)

                    # Labeling
                    label = 1 if np.max(slice_SEG) > 0 else 0

                    patient_images.append(three_channel_tensor)
                    patient_labels.append(label)

        except Exception as e:
            print(f"Skipping patient {patient_folder} due to file error: {e}")
            continue

        # Save Patient Data and Clean RAM
        if patient_images:
            X_patient = np.array(patient_images, dtype=np.float32)
            Y_patient = np.array(patient_labels, dtype=np.int32)
            
            # Use savez_compressed for efficient disk storage
            np.savez_compressed(file_path, X=X_patient, Y=Y_patient) 
            saved_file_paths.append(file_path)
            
            # CRUCIAL: Delete arrays/lists to free RAM
            del patient_images, patient_labels, X_patient, Y_patient 

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(patient_folders)} patients. Total files: {len(saved_file_paths)}")

    print("-" * 50)
    print(f"Data processing complete. {len(saved_file_paths)} patient data files saved to disk.")
    return saved_file_paths

# --- PYTORCH DATASET WRAPPER ---

class MRIDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long() 
        self.features = self.features.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    
    # 1. Check for final output file (MODIFIED)
    if os.path.exists(FULL_OUTPUT_FILE):
        print(f"FINAL FULL DATA FOUND. Skipping all processing. Use '{FULL_OUTPUT_FILE}' for subsequent steps.")
        exit()

    # 2. Setup output directory
    if not os.path.exists(OUTPUT_DATA_DIR):
        os.makedirs(OUTPUT_DATA_DIR)
        print(f"Created output directory: {OUTPUT_DATA_DIR}")
    
    # 3. Load/Process data or list existing files
    patient_data_files = glob(os.path.join(OUTPUT_DATA_DIR, "*_data.npz"))
    
    if len(patient_data_files) > 0:
        print(f"Found {len(patient_data_files)} intermediate patient files. Skipping NIfTI processing.")
    else:
        patient_data_files = load_and_preprocess_data(DATA_PATH, OUTPUT_DATA_DIR) 
    
    if not patient_data_files:
        print("Script terminated. No data files were found or processed.")
        exit()

    # 4. CONSOLIDATE ALL SAVED PATIENT FILES VIA MEMORY-MAP (Final Memory Fix)

    print("-" * 50)
    print("Consolidating all saved patient data into a memory-mapped array (Disk-backed)...")

    # Determine the total size (N) without loading all data into RAM
    total_samples = 0
    for file_path in patient_data_files:
        with np.load(file_path) as data:
            total_samples += data['X'].shape[0]

    # Use paths relative to the current directory for memmap files
    X_final_path = 'brats_cnn/X_final_memmap.dat'
    Y_final_path = 'brats_cnn/Y_final_memmap.dat'
    
    # Create the memory-mapped arrays on disk
    X_memmap = np.memmap(X_final_path, dtype='float32', mode='w+', 
                         shape=(total_samples, 64, 64, 3))
    Y_memmap = np.memmap(Y_final_path, dtype='int32', mode='w+', 
                         shape=(total_samples,))

    # Write data sequentially into the memmap files
    current_idx = 0
    for file_path in patient_data_files:
        with np.load(file_path) as data:
            X_patient = data['X']
            Y_patient = data['Y']
        
        n_samples = X_patient.shape[0]
        
        # Write the patient data directly to the disk space managed by the memmap object
        X_memmap[current_idx:current_idx + n_samples] = X_patient
        Y_memmap[current_idx:current_idx + n_samples] = Y_patient
        
        current_idx += n_samples
        del X_patient, Y_patient 

    X_memmap.flush() 
    Y_memmap.flush()
    print(f"Consolidation complete. Total Samples: {total_samples}.")

    # 5. SAVE FULL DATASET (MODIFIED - NO SPLIT)
    
    # Save the entire content of the memory-mapped arrays to a single compressed file
    np.savez_compressed(FULL_OUTPUT_FILE, X_full=X_memmap, Y_full=Y_memmap)
    
    # Clean up memmap files
    del X_memmap, Y_memmap 
    os.remove(X_final_path) 
    os.remove(Y_final_path)

    print("-" * 50)
    print("SUCCESS: Full Dataset processing complete!")
    print(f"Total Samples Saved: {total_samples}")
    print(f"Full dataset saved to {FULL_OUTPUT_FILE}.")
    print("-" * 50)