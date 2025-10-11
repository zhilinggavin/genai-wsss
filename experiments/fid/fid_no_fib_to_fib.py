import torch
from pytorch_fid import fid_score
import os
from tqdm import tqdm
gpu_num = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num

'''
For each set of manipulated images, calculate the FID score against the opposite original images.
'''
real_images_path = '/media/NAS06/gavinyue/disentanglement/scripts_segmentation/result_exp/fid/fibrosis/fid_dataset/orig'
# real_images_path = '/media/NAS06/gavinyue/disentanglement/scripts_segmentation/result_exp/fid/no_fibrosis/fid_dataset/orig_covid'


# for i in tqdm(range(1,13)):
#     ms = i * 0.25
for ms in [1.25,1.50]:    
    # generated_images_path = f'/media/NAS06/gavinyue/disentanglement/scripts_segmentation/result_exp/fid/no_fibrosis/fid_dataset/manip{ms:.2f}'
    generated_images_path = f'/media/NAS06/gavinyue/disentanglement/scripts_segmentation/result_exp/fid/no_fibrosis/fid_dataset/cov_manip{ms:.2f}'
    fid_value = fid_score.calculate_fid_given_paths(
        [real_images_path, generated_images_path],
        batch_size=50,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        dims=2048
    )

    print(f'Manipulation: {ms:.2f}, FID Score: {fid_value}')

    save_path = os.path.join(os.path.dirname(generated_images_path), 'fid_score.txt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Write FID Score to a file
    with open(save_path, 'a') as file:
        if ms == 1.25:
            file.write(f'\nreal_images_path: {real_images_path}\nFID Score Bellow:\n')
        file.write(f'manip{ms:.2f}, FID Score: {fid_value}\n')
