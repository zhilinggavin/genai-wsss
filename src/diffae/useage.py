'''
Usage and Template of diffusion autoencoder model
'''
import os
import numpy as np
import torch
import torch.nn as nn


from src.diffae.templates import ffhq256_autoenc
from src.diffae.experiment import LitModel

import logging
logging.basicConfig(level=logging.INFO)

def model_load(device: str, diff: bool = True, cls: bool = True):
    '''
    Load ISBI diff model and classification model
    '''
    # sys.path.append("/media/NAS06/gavinyue/disentanglement/scripts_segmentation")
    model_diff = model_cls = None
    root_dir = '/media/NAS_R02/USER_PATH/gavinyue/genai-wsss'
    if diff:
        # from templates import ffhq256_autoenc #type: ignore
        # from experiment import LitModel #type: ignore
        conf = ffhq256_autoenc()
        model_diff = LitModel(conf)
        # conf.name = 'test_osic256_cluster_mxl_round2'
        state = torch.load(f'{root_dir}/experiments/wsss_unet/checkpoints/diffusion/isbi_checkpoints/last.ckpt', map_location='cpu')
        model_diff.load_state_dict(state['state_dict'], strict=False)
        model_diff.ema_model.eval()
        model_diff.ema_model.to(device)
        logging.info('ISBI diffusion model loaded')
    if cls:
        # load classification model
        model_cls = nn.Linear(512, 2)
        model_cls = model_cls.to(device)
        state_dict = torch.load(f'{root_dir}/experiments/wsss_unet/checkpoints/classifier/isbi_cls_loss_best_499.pt')['model_state_dict']
        
        
        if any(key.startswith('module.') for key in state_dict.keys()):
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model_cls = nn.DataParallel(model_cls)

            model_cls.load_state_dict(state_dict)
            model_cls.direction_class_0 = model_cls.module.weight[0]
            model_cls.direction_class_1 = model_cls.module.weight[1]
        else:
            model_cls.load_state_dict(state_dict)
            model_cls.direction_class_0 = model_cls.weight[0]
            model_cls.direction_class_1 = model_cls.weight[1]


        logging.info('ISBI classification model loaded')
        # setattr(model_cls, 'direction_class_0', model_cls.weight[0].detach().cpu().numpy())
        # setattr(model_cls, 'direction_class_1', model_cls.weight[1].detach().cpu().numpy())
    return model_diff, model_cls

def encode_latent(model, x: torch.Tensor, device: str):
    '''
    Usage Example: Encode image to latent code using diffae model
    '''
    x = x.to(device)
    with torch.no_grad():
        model.eval()
        cond = model.encode(x)
    return cond