#!/usr/bin/env /home/gavinyue/miniconda3/envs/codebase/bin/python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from CBFNet import CBFNet
import numpy as np
import yaml
from networks import *
from utils import *
from glob import glob
from collections import defaultdict
from tqdm import tqdm
from findContours import water_osic
from PIL import Image 
# import matplotlib.pyplot as plt

config_path = "./configs/OSIC_default.yaml"
# Load the YAML file
with open(config_path, 'r') as file:
    args = yaml.unsafe_load(file)
args.exp_name = 'less_equal_trainA' # the trainset is set equal to no_fibrosis, around 2000 images
print(args)  # Output: Namespace(phase='train', batch_size=32).
# debug orig pre-trained model
# args.dataset = 'BraTS'; args.folder = 'brats_1'

gan = CBFNet(args)
gan.build_model()

# load model
def load_model(self: CBFNet, iter: str, model_path: str = None):
    # Load the model weights
    if model_path is None:        
        iter = '0009000'
        model_path = glob(
                os.path.join(self.result_dir, self.dataset, self.folder, self.exp_name, f"*{iter}*"))
        assert len(model_path) == 1, f"Model path not found for iteration {iter}"
        model_path = model_path[0]
    self.model_name = str(model_path.split('/')[-1].split('\'')[0])
    params = torch.load(model_path)
    self.genA2B.load_state_dict(params['genA2B'])
    self.genB2A.load_state_dict(params['genB2A'])
    self.disGA.load_state_dict(params['disGA'])
    self.disGB.load_state_dict(params['disGB'])
    self.disLA.load_state_dict(params['disLA'])
    self.disLB.load_state_dict(params['disLB'])
    
    print(f"Model loaded from {model_path}")
    
    self.path_seg = os.path.join(self.result_dir, self.dataset, self.folder, 'testA_folder',
                'pred_mask_0for4', self.model_name)
    os.makedirs(self.path_seg, exist_ok=True)

    return self

iter = '0009000'
load_model(gan, iter)


def metric(pred: np.ndarray, gt: np.ndarray):
    from torchmetrics.classification import Dice, BinaryJaccardIndex
    dice_metric = Dice(num_classes=1, multiclass=False, ).to('cuda')
    iou_metric = BinaryJaccardIndex().to('cuda') # Dice = 2*IoU / (IoU + 1)
    if pred.shape[-1] == 3:
        pred = pred[..., 0]
    if gt.max() > 1:
        gt = (gt / 255).astype(np.uint8)
    if pred.max() > 1:
        pred = (pred / 255).astype(np.uint8)
    pred = torch.from_numpy(pred).to('cuda')
    gt = torch.from_numpy(gt).to('cuda')

    dice_metric.reset()
    iou_metric.reset()
    dice = dice_metric(pred, gt)
    jaccard = iou_metric(pred, gt)
    # print(f"Dice: {dice:4f}, Jaccard: {jaccard:4f}")

    return dice.detach().cpu().numpy(), jaccard.detach().cpu().numpy()

def get_metrics(imgs):
    # dice, iou = metric(imgs['seg'], imgs['gt'])
    dice_scores = []
    iou_scores = []
    for seg, gt in zip(imgs['seg'], imgs['gt']):
        dice, iou = metric(seg, gt)
        dice_scores.append(dice)
        iou_scores.append(iou)

    # save to csv
    import pandas as pd
    df = pd.DataFrame({
        'name': imgs['name'],
        'dice': dice_scores,
        'iou': iou_scores,
    })
    df.to_csv(os.path.join(gan.path_seg, 'metrics.csv'), index=False)
    print(f"Metrics saved to {gan.path_seg}/metrics.csv")


    # Compute average metrics
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    print(f"Average Dice: {avg_dice}, Average IoU: {avg_iou}")

    median_dice = np.median(dice_scores)
    median_iou = np.median(iou_scores)
    print(f"Median Dice: {median_dice}, Median IoU: {median_iou}")
            
def infer(self: CBFNet):
    self.genA2B.eval(), self.genB2A.eval()
    imgs = defaultdict(list)
    names = []
    for n, (real_A, gt, pathA) in enumerate(tqdm(self.testA_loader)):
        real_A = real_A.to(self.device)
        real_A2 = real_A

        fake_A2B, cam_logit, fake_A2B_heatmap, att_maskA2B, fake_A2B_2, real_A2_r = self.genA2B(real_A2)
        pathA = pathA[0].split('/')[-1].split('.')[0] + '.' + pathA[0].split('/')[-1].split('.')[1]
        
        
        background = torch.zeros_like(att_maskA2B).to(self.device)
        att_maskA2B = torch.where(real_A2 > -1, att_maskA2B, background)
        
        if (self.dataset == "OSIC"):
            att_maskA2B = norm_01(att_maskA2B ** 5)
            pathA = pathA.split('.')[0] #remove '.png'

        att_maskA2B_r = 1 - att_maskA2B
        
        REAL_A = RGB2BGR(tensor2numpy(denorm(real_A[0])))
        MASKA2B = RGB2BGR(tensor2numpy(denorm(att_maskA2B[0])) * 2 - 1)
        MASKA2B_R = RGB2BGR(tensor2numpy(denorm(att_maskA2B_r[0])) * 2 - 1)


        HEATMAP_A2B = cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size)
        FAKEA2B = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))
        FAKEA2B_2 = RGB2BGR(tensor2numpy(denorm(fake_A2B_2[0])))
        
        # Postprocessing: get final segmentation
        seg = water_osic((MASKA2B_R*255).astype(np.uint8))
        Image.fromarray(seg).save(os.path.join(self.path_seg, pathA + '.png'))
        
        # append imgs
        imgs['real_A'].append(REAL_A*255)
        # imgs['maskA2B'].append(MASKA2B*255)
        # imgs['maskA2B_r'].append(MASKA2B_R*255)
        # imgs['fake_A2B'].append(FAKEA2B*255)
        # imgs['fake_A2B_2'].append(FAKEA2B_2*255)
        # imgs['heatmap_A2B'].append(HEATMAP_A2B*255)
        
        imgs['seg'].append(seg[:,:,0])
        imgs['gt'].append(gt[0].cpu().numpy())
        names.append(pathA)
        
        # if n >= 20:
        #     break
        
    imgs = {keys: np.array(imgs[keys]).astype(np.uint8) for keys in imgs.keys()}
    imgs['name'] = names
    
    # return imgs
    get_metrics(imgs)

infer(gan)
