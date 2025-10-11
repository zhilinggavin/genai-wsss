'''
This code is for training classifer of fibrosis images.
Author: Gavin Yue, modified from Yingying Fang
'''

# import sys
# sys.path.append("..")
# gpu_num = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
# from utils import quality_check
# from select_dataloader import *
# from select_model import *
# from select_optimizer import *
# from select_loss import *
# from sklearn.model_selection import StratifiedKFold
#
# from select_parameters import pld_mortality_train_parameter, write_parameter
# from torch.utils.data import DataLoader
# from trainer import *


# import wandb
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
# from train_10montage import *
import torch
import numpy as np
import os
import random
from torchvision import transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import sys
sys.path.append("..")
import stylegan_codebase.fns_custom.fn_stylegan_oisc as fns
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

# opt = pld_mortality_train_parameter()
#

# save_location = '../results/result_cls_fibrosis/'
# save_logdir = save_location + opt.expname + '/logdir/'  # fold1/2/3/4/5
# save_model = save_location + opt.expname + '/model/'  # fold1/2/3/4/5, best_auc, best_loss
# save_param = save_location + opt.expname + '/para.txt'
# save_result = './result_exp/' + opt.expname + '/result.csv'
#
# os.makedirs(save_logdir, exist_ok=True)
# os.makedirs(save_model, exist_ok=True)
# write_parameter(opt, save_param)
# writer = SummaryWriter(log_dir=save_logdir)
#

''' exp1_moredata: ratio of fib: no_fib = 3.55:1 '''
# expname = 'test2_moredata'
# trainset_loader, testset_loader = fns.prepare_dataset(bs=8, dataname='all_data', train_mode=True)

''' exp2_equal_ratio: ratio of fib: no_fib = 1:1 '''

''' exp3_equal_ratio: ratio of fib: no_fib = 1:1 '''
expname = 'exp3_equal_ratio'
trainset_loader, testset_loader = fns.prepare_dataset(bs=8, dataname='all_data', train_mode=True, exp = expname)


device = "cuda"
# Traing model from scratch
model = nn.Linear(512, 2)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)
# consume training data
cls_model_path = '/media/NAS06/gavinyue/disentanglement/scripts_segmentation/result_exp/classification_fibrosis/exp3_equal_ratio/model/model_loss_best_162.pt'
state_dict = torch.load(cls_model_path)['model_state_dict']
model.load_state_dict(state_dict)


model_diff,_ = fns.model_load(device,diff=True,cls=False)

lossfun = nn.CrossEntropyLoss()

lr = 1e-6 #learning rate
optimizer = optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
epochs = 500 #300

# Use tensorboard to visualize the training process.

save_result = './result_exp/classification_fibrosis/' + expname
save_model = save_result + '/model/'
save_log = os.path.join(save_result, 'log')
os.makedirs(save_log, exist_ok=True)
os.makedirs(save_model, exist_ok=True)

writer = SummaryWriter(log_dir= save_log)

import wandb
wandb.login(key="7e1cc56e331175f1cd23f8e739a7fbff0783cf6e")
# wandb.init(project='classifer_fibrosis', name='exp3')
wandb.init(project='classifer_fibrosis', name='exp3', id='4uat5swk', resume="allow")
print("wandb run ID:", wandb.run.id) #4uat5swk


total_loss_test_best = 1000
total_acc_test_best = 0.1
total_acu_test_best = 0.1

iteration = 0
for epoch in tqdm(range(162, epochs)):

    # print(epoch)
    model.train()

    list_label_train = []
    list_pred_train  = []
    list_score_train = []
    total_loss_train = 0
    for image_axial, labels, names in trainset_loader:

        iteration += 1
        image_axial, labels = image_axial.to(device), labels.to(device)
        image_diff = model_diff.encode(image_axial.to(device))

        optimizer.zero_grad()
        outputs = model(image_diff)

        loss_fuse_ob = lossfun(outputs.float(), labels.long())
        loss_fuse_ob.backward()

        total_loss_train += loss_fuse_ob.item()
        optimizer.step()

        pred = F.softmax(outputs, dim=1).reshape(outputs.size()[0], -1)
        score = pred.data[:, 1]
        _, pred_class = torch.max(outputs, dim=1)

        list_label_train.extend(labels.tolist())
        list_pred_train.extend(pred_class.tolist())
        list_score_train.extend(score.tolist())

        # if iteration % 500 == 0:
        #     print(loss_fuse_ob.item())

    acc_train = accuracy_score(list_label_train, list_pred_train)
    cfm_train = confusion_matrix(list_label_train, list_pred_train, labels=[0,1])
    spec_train = cfm_train[0][0] / np.sum(cfm_train[0]) # Specificity = TN / (TN + FP)
    sens_train = cfm_train[1][1] / np.sum(cfm_train[1]) # Sensitivity = TP / (TP + FN)
    try:
        auc_train = roc_auc_score(list_label_train, list_score_train)
    except:
        auc_train = 0
    writer.add_scalar('train/loss', total_loss_train, epoch)
    writer.add_scalar('train/auc',  auc_train, epoch)
    writer.add_scalar('train/acc',  acc_train, epoch)
    writer.add_scalar('train/sens', sens_train, epoch)
    writer.add_scalar('train/spec', spec_train, epoch)
    
    # Log the metrics in wandb cloud
    wandb.log({'train/loss': total_loss_train, 'epoch': epoch})
    wandb.log({'train/auc': auc_train, 'epoch': epoch})
    wandb.log({'train/acc': acc_train, 'epoch': epoch})
    wandb.log({'train/sens': sens_train, 'epoch': epoch})
    wandb.log({'train/spec': spec_train, 'epoch': epoch})


    # model.val()
    list_label_test = []
    list_pred_test  = []
    list_score_test = []
    total_loss_test = 0
    for image_axial, labels, names in testset_loader:
        image_axial, labels = image_axial.to(device), labels.to(device)
        image_diff = model_diff.encode(image_axial.to(device))

        optimizer.zero_grad()
        outputs = model(image_diff)

        loss_fuse_ob = lossfun(outputs.float(), labels.long())
        total_loss_test += loss_fuse_ob.item()

        pred = F.softmax(outputs, dim=1).reshape(outputs.size()[0], -1)
        score = pred.data[:, 1]
        _, pred_class = torch.max(outputs, dim=1)

        list_label_test.extend(labels.tolist())
        list_pred_test.extend(pred_class.tolist())
        list_score_test.extend(score.tolist())

        # if iteration % 500 == 0:
        #     print(loss_fuse_ob.item())


    acc_test = accuracy_score(list_label_test, list_pred_test)
    cfm_test = confusion_matrix(list_label_test, list_pred_test, labels=[0,1])
    spec_test = cfm_test[0][0] / np.sum(cfm_test[0])
    sens_test = cfm_test[1][1] / np.sum(cfm_test[1])
    try:
        auc_test = roc_auc_score(list_label_test, list_score_test)
    except:
        auc_test = 0
    writer.add_scalar('test/loss', total_loss_test, epoch)
    writer.add_scalar('test/auc',  auc_test, epoch)
    writer.add_scalar('test/acc',  acc_test, epoch)
    writer.add_scalar('test/sens', sens_test, epoch)
    writer.add_scalar('test/spec', spec_test, epoch)
    
    # Log the metrics in wandb cloud
    wandb.log({'test/loss': total_loss_test, 'epoch': epoch})
    wandb.log({'test/auc': auc_test, 'epoch': epoch})
    wandb.log({'test/acc': acc_test, 'epoch': epoch})
    wandb.log({'test/sens': sens_test, 'epoch': epoch})
    wandb.log({'test/spec': spec_test, 'epoch': epoch})


    if total_loss_test < total_loss_test_best:
        total_loss_test_best = total_loss_test
        spec_test_best = spec_test
        sens_test_best = sens_test
        if epoch>100:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict()},
                    # 'optimizer_state_dict': optimizer.state_dict()},
                    save_model + 'model_loss_best_'+ str(epoch) + '.pt')

        # print("Sens: {}, Spec: {}".format(auc_test, acc_test, sens_test_best, spec_test_best))
        print("Best loss:\n AUC: {}, ACC: {}, Sens: {}, Spec: {}, Loss: {}".format(auc_test, acc_test, sens_test_best, spec_test_best, total_loss_test))
        
    # Modified by Gavin Yue
    if auc_test > total_acu_test_best:
        total_acu_test_best = auc_test
        if epoch>100:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict()},
                    save_model + 'model_auc_best_'+ str(epoch) + '.pt')
        print("Best auc:\n AUC: {}, ACC: {}, Sens: {}, Spec: {}, Loss: {}".format(auc_test, acc_test, sens_test, spec_test, total_loss_test))
        
    if acc_test > total_acc_test_best:
        total_acc_test_best = acc_test
        if epoch>100:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict()},
                    save_model + 'model_acc_best_'+ str(epoch) + '.pt')

        print("Best acc:\n AUC: {}, ACC: {}, Sens: {}, Spec: {}, Loss: {}".format(auc_test, acc_test, sens_test, spec_test, total_loss_test))

writer.close()
