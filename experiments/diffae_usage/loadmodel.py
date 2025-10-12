import os
import sys
sys.path.append(os.getcwd())
from utils.diffae.templates import ffhq256_autoenc

import torch
import torch.nn as nn

def model_load(device, diff=True, cls=True):
    '''
    Load OLD diff model and OLD classification model, trained by Yingying Fang.
    '''
    # sys.path.append("/media/NAS06/gavinyue/disentanglement/scripts_segmentation")
    model_diff = model = None
    if diff:
        from templates import ffhq256_autoenc #type: ignore
        from experiment import LitModel #type: ignore
        conf = ffhq256_autoenc()
        model_diff = LitModel(conf)
        conf.name = 'test_osic256_cluster_mxl_round2'
        state = torch.load(f'/media/NAS04/yyfang/prognostic_result/xai/counterfactual/Diffusion-Explainer/checkpoints/{conf.name}/last.ckpt', map_location='cpu')
        model_diff.load_state_dict(state['state_dict'], strict=False)
        model_diff.ema_model.eval()
        model_diff.ema_model.to(device)
        print('OLD model diff loaded')
    if cls:
        # load classification model
        model = nn.Linear(512, 2)
        model = model.to(device)
        state_dict = torch.load('/media/NAS04/yyfang/prognostic_result/xai/counterfactual/Diffusion-Explainer/scripts_osic/result_exp/classification_fibrosis/test/model/model_loss_best.pt')['model_state_dict']
        model.load_state_dict(state_dict)
        print('OLD classification model loaded')

        # self.direction_class_0 = model.weight[0].detach().cpu().numpy()
        # self.direction_class_1 = model.weight[1].detach().cpu().numpy()
    return model_diff, model