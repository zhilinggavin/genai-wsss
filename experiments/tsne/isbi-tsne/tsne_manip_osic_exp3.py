# %%
'''
TSNE visualization of original data and manipulated data
Date: 2024-07-11
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
import sys
sys.path.append("/media/NAS06/gavinyue/disentanglement")
import stylegan_codebase.fns_custom.fn_stylegan_oisc as fns
from tqdm import tqdm

''' exp3_equal_ratio: ratio of fib: no_fib = 1:1 '''
expname = 'exp3_equal_ratio'

device = "cuda"
max_count0 = 336 #@params[30,236,336,len(trainset_loader)-1,'alldata']
max_count1 = 30 #@params[12,30]
cond_fib, cond_nofib = fns.load_orig_data_tsne(device,max_count=max_count0,exp=expname)


'''
Load manipulated and encoded data
'''
def load_orig_data_tsne(device,max_count=30,exp='exp3_equal_ratio'):
    # max_count#@params[30,236,len(trainset_loader)-1] 
    M = OsicMask()
    trainset_loader =prepare_dataset(bs=8, dataname='fibrosis',exp=exp)
    if max_count == 'alldata':
        count = len(trainset_loader)-1
    else:
        count = max_count
        print(f'{count}')
    # M.setup_osic(trainset_loader, device, max_count=count, model_load = False, save_encode = False)
    # cond_fib = M.img_cond
    cond_fib = np.load(f'/media/NAS06/gavinyue/disentanglement/scripts_segmentation/npy/osic/manip_shuffled/fibrosis/{exp}/cond/recon/count{count}.npy')

    trainset_loader =prepare_dataset(bs=8, dataname='no_fibrosis',exp=exp)
    if max_count == 'alldata':
        count = len(trainset_loader)-1
    else:
        count = max_count
    # M.setup_osic(trainset_loader, device, max_count=count, model_load = False, save_encode = False)
    # cond_nofib = M.img_cond
    cond_nofib = np.load(f'/media/NAS06/gavinyue/disentanglement/scripts_segmentation/npy/osic/manip_shuffled/no_fibrosis/{exp}/cond/recon/count{count}.npy')

    cond_fib = np.concatenate(cond_fib)
    cond_nofib = np.concatenate(cond_nofib)
    return cond_fib, cond_nofib



cond_nofib = cond_nofib[max_count1*8:]

# manip_idx=4 #@params[0,1,2,3,4]
for manip_idx in [0,1,2,3,4]:
    # alpha = (2*manip_idx + 1)
    alpha = manip_idx * 3/4
    
    cond_nofib_manip,model = fns.load_encode_img(device=device,dataname='no_fibrosis',max_count=max_count1,manip_idx=manip_idx,cls=True,exp=expname)
    cond_nofib_minus_manip,_ = fns.load_encode_img(device=device,dataname='no_fibrosis_minus',max_count=max_count1,manip_idx=manip_idx,exp=expname)
    if model is not None:
        try:
            weights = model.weight.detach().cpu().numpy()
        except:
            weights = model.module.weight.detach().cpu().numpy()
        direction_class_0, direction_class_1 = weights[0], weights[1]            

    cond_nofib_manip = np.concatenate(cond_nofib_manip)
    cond_nofib_minus_manip = np.concatenate(cond_nofib_minus_manip)

    embedded_nofib = np.concatenate([cond_fib,cond_nofib,cond_nofib_manip])
    embedded_nofib_minus = np.concatenate([cond_fib,cond_nofib,cond_nofib_minus_manip])
    
    # embedded_nofib = np.concatenate([cond_fib,cond_nofib])
    # embedded_nofib_minus = np.concatenate([cond_fib,cond_nofib])
    
    if model is not None:
        embedded_nofib = embedded_nofib*direction_class_1
        embedded_nofib_minus = embedded_nofib_minus*direction_class_1
    
    a1 = len(cond_fib); a2 = len(cond_nofib); a3 = len(cond_nofib_manip)
    print(f'Fibrosis: {a1}, No_Fibrosis: {a2}, Manip: {a3}, all: {a1+a2+a3}')
    # Larger datasets usually require a larger perplexity, normally [5,50]. 
    # A reasonable starting point might be around the square root of the number of data points
    perp_assume = int(np.sqrt(a1+a2+a3)/5)*5
    

    tsne = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=perp_assume, random_state=20)

    embedded_nofib_tsne = tsne.fit_transform(embedded_nofib)
    embedded_nofib_minus_tsne = tsne.fit_transform(embedded_nofib_minus)

    # Plot the embedded data
    tsne_data = [(embedded_nofib_tsne, f'No_Fib_Add(a={alpha})'), (embedded_nofib_minus_tsne, f'No_Fib_Minus(a={alpha})')]
    fig, axs = plt.subplots(1, len(tsne_data), figsize=(8*len(tsne_data), 8),dpi=150)

    for ax, (data, title) in zip(axs, tsne_data):
        ax.scatter(data[:a1, 0], data[:a1, 1], color='blue', label=f'Fibrosis({a1})', s=5)
        ax.scatter(data[a1:a1+a2, 0], data[a1:a1+a2, 1], color='green', label=f'No_Fibrosis({a2})', s=5)
        ax.scatter(data[a1+a2:, 0], data[a1+a2:, 1], color='red', label=f'Manipulated({a3})', s=10)
        ax.set_title(title)
        ax.legend(loc='upper right')
    if model is not None:
        tmp = f'/media/NAS06/gavinyue/disentanglement/scripts_segmentation/result_exp/t-sne/manip/no_fibrosis_to_fib/{expname}/count{max_count1}/weight1'
    else:
        tmp = f'/media/NAS06/gavinyue/disentanglement/scripts_segmentation/result_exp/t-sne/manip/no_fibrosis_to_fib/{expname}/count{max_count1}/no_weight'
    os.makedirs(tmp,exist_ok=True)
    # plt.savefig(f'{tmp}/perplexity{perp_assume}_alpha{alpha}_separate.jpg')
    plt.savefig(f'{tmp}/{expname}_alpha{alpha}.jpg')
    plt.close('all')