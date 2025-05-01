import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
import pandas as pd

def load_slices(img_dir, caseid, img_dir2 = None):
    """
    Load 2D image slices from a folder and stack them into a 3D NumPy array.
    Returns:
        np.ndarray: 3D volume array (Z, H, W).
    """
    caseid = caseid.zfill(3)  # Ensure case ID is zero-padded to 3 digits
    file_names = sorted(os.listdir(img_dir))
    
    if img_dir2:
        file_names2 = os.listdir(img_dir2)
        file_names = sorted(file_names + file_names2)
    
    file_names = [name for name in file_names if name.startswith(caseid) and name.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    slices = []
    
    for fname in file_names:
        try:
            img = Image.open(os.path.join(img_dir, fname)).convert('L')  # Convert to grayscale
        except FileNotFoundError:
            img = Image.open(os.path.join(img_dir2, fname)).convert('L')

        slices.append(np.array(img))

    volume = np.stack(slices, axis=0)  # shape: (Z, H, W)
    return volume


def convert_to_nifti(volume_array, save_path):
    """
    Convert a 3D NumPy array (Z, H, W) to NIfTI format and save it.
    """
    # Convert NumPy array to SimpleITK Image
    volume_sitk = sitk.GetImageFromArray(volume_array)  # assumes Z, Y, X

    # Optional: set spacing (e.g., from original scan metadata)
    volume_sitk.SetSpacing([1.0, 1.0, 1.0])  # [X, Y, Z] spacing in mm

    # Save as .nii.gz
    sitk.WriteImage(volume_sitk, save_path)
    
def save_as_nii(caseid, save_path, img_dir, img_dir2 = None):
    """
    Save the 2D slices as a 3D NIfTI file.
    """
    volume_array = load_slices(img_dir, caseid, img_dir2)
    convert_to_nifti(volume_array, save_path)
    print(f"Saved {caseid} to {save_path}")

    
if __name__ == "__main__":
    # Directories for 2D slices
    img_dir = '../data/OSIC/processed/fibrosis'
    img_dir2 = '../data/OSIC/processed/no_fibrosis'
    save_dir = '../data/OSIC/processed/3d_visual'
    os.makedirs(save_dir, exist_ok=True)
    
    # Load case IDs from CSV
    df = pd.read_csv('../data/OSIC/doctor_category.csv')
    caseids_test = df['case_id'][df['test'] == 1].tolist()
    caseids_test = [str(caseid).zfill(3) for caseid in caseids_test]
    print(f"Processing case IDs: {caseids_test}")
    
    # Process each case ID
    for caseid in caseids_test:
        save_path = os.path.join(img_dir, caseid + '_recon.nii.gz')
        save_as_nii(caseid, save_path, img_dir, img_dir2)