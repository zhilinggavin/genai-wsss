import os, glob
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    # Specify the root directory
    root_dir = "../data/YYF_30Case" # Replace with your folder path
    # sub_dir = "labels_1"
    sub_dir = "labels_2"
    img_path = os.path.join(root_dir, sub_dir)
    assert os.path.exists(root_dir), f"Error: The directory '{root_dir}' does not exist."
    assert os.path.exists(img_path), f"Error: The directory '{img_path}' does not exist."
    
    files = os.listdir(img_path)
    # Sort the files based on their left numeric value
    cts = [file for file in files if file.endswith('.nii.gz')]
    cts = sorted(cts)
    
    '''
    load all cts
    '''
    count = 0
    for ct_name in tqdm(cts):
        print(f'count: {count}, ct_name: {ct_name}')
        count += 1

        ct_512_orig = os.path.join(img_path, ct_name)
        # ct_512_fibrosis = os.path.join(path, ct_name)
        
        try:
            ct_512_orig_stik = sitk.ReadImage(ct_512_orig) #shape (slices, 512, 512)
        except:
            print("CT file read failed!")
            continue
        
        ct_512_orig_np = sitk.GetArrayFromImage(ct_512_orig_stik)
        dir_name = img_path.replace(sub_dir, sub_dir+'_imgs')
        save_folder = os.path.join(dir_name, ct_name.replace('.nii.gz',''))
        os.makedirs(save_folder, exist_ok=True)
        
        for z in range(ct_512_orig_np.shape[0]):
            slice_np = ct_512_orig_np[z, :, :]
            slice_img = Image.fromarray((slice_np * 255).astype(np.uint8))
            
            imgname = ct_name.replace('.nii.gz', f'_{z:03d}.png')
            save_path = os.path.join(save_folder, imgname)
            slice_img.save(save_path)
        print(f"Saved {ct_name} to {save_folder}")
        # break