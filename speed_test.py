import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" 
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="requests")
import argparse
import sys
import math
import copy

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
from omegaconf import OmegaConf

from protego.utils import get_usable_img_paths
from protego.protego_train import train_protego_mask
from protego.protego_train_lpips import train_protego_mask_lpips
from protego.chameleon_train import train_chameleon_mask
from protego.opom_train import train_opom_mask
from protego.run_exp import run
from protego.FacialRecognition import BASIC_POOL, SPECIAL_POOL
from protego import BASE_PATH

if __name__ == "__main__":
    ####################################################################################################################
    # Configuration
    ####################################################################################################################
    args = argparse.ArgumentParser()
    args.add_argument('--exp_name', type=str, default='simple_cosine', help='The name of the experiment.')
    args.add_argument('--device', type=str, help='The device to use for training. (cpu, mps, cuda:0, etc.)')
    args = args.parse_args()

    configs = {
        # Running env
        'global_random_seed' : 42, 
        'device': 'cuda:0',  
        'exp_name': 'default', 
        
        # Training data
        'train_portion': 0.6, 
        'uv_gen_align_ldmk': False, 
        'uv_gen_batch': 8, 
        'need_cropping': False, 
        'fd_name': 'mtcnn', 
        'crop_loosen': 1., 
        'shuffle': False, 

        # Training configs
        'three_d': True,
        'epoch_num': 100,  
        'batch_size' : 4, 
        'epsilon' : 16 / 255., 
        'min_ssim' : 0.95, 
        'learning_rate' : 0.01 * (16 / 255.), # Protego and Chameleon: 0.01 * (16 / 255.) OPOM: 1 / 255.
        'mask_size' : 224, 
        'mask_random_seed': 114, 
        'bin_mask': True, # Whether to use binary mask. If True, the perturbation will be restricted to the face area. 
        'train_fr_names': [n for n in BASIC_POOL if n != 'ir50_adaface_casia'],  

        # Eval configs
        'eval_scene': 1,
        'mask_name': ['default', 'univ_mask.npy'], 
        'eval_db': 'face_scrub',
        'eval_fr_names': ['ir50_adaface_casia'],
        'save_univ_mask': True, 
        'visualize_interval': 10,
        'query_portion': 0.5,
        'vis_eval': True, 
        'lpips_backbone': "vgg", 
        'end2end_eval': False, 
        'strict_crop': True, 
        'smoothing': 'none', # Options: 'gaussian', 'median', 'none'
        'resize_face': True, 
        'jpeg': False,
        'eval_compression': False, # Whether to evaluate the compression of the mask.
        'eval_compression_methods': ['gaussian', 'median', 'jpeg', 'resize'], # The compression methods to evaluate.
        'compression_cfgs' : {
            # Gaussian Filter
            'gaussian': {
                'kernel_size': 9, 
                'sigma': 2.0,
            }, 
            # Median Filter
            'median': {
                'kernel_size': 9
            }, 
            # JPEG Compression
            'jpeg': {
                'quality': 70, 
            }, 
            # Resize
            'resize': {
                'resz_percentage': 0.4,  # Resize the image to 50% of its original size
                'mode': 'bicubic'
            }
        }
    }
    ####################################################################################################################
    # Run
    ####################################################################################################################
    cfgs = OmegaConf.create(configs)
    cfgs.exp_name = args.exp_name if '--exp_name' in sys.argv else cfgs.exp_name
    cfgs.device = args.device if '--device' in sys.argv else cfgs.device
    torch.manual_seed(cfgs.global_random_seed)

    train_portion = cfgs.train_portion
    shuffle_data = cfgs.shuffle
    train_data_path = os.path.join(BASE_PATH, 'face_db', 'face_scrub')
    protectees = sorted([name for name in os.listdir(train_data_path) if not name.startswith(('.', '_'))])
    data = {}
    for protectee in protectees:
        protectee_path = os.path.join(train_data_path, protectee)
        imgs = get_usable_img_paths(protectee_path)
        train_num = math.floor(len(imgs) * train_portion)
        eval_num = len(imgs) - train_num
        if shuffle_data:
            rand_gen = torch.Generator()
            rand_gen.manual_seed(cfgs.global_random_seed)
            indices = torch.randperm(len(imgs), generator=rand_gen).tolist()
            imgs = [imgs[i] for i in indices]
        data[protectee] = {'train': imgs[:train_num], 'eval': imgs[train_num:]}
    #run(cfgs, mode='train', data=data, train=train_protego_mask)
    #run(cfgs, mode='train', data=data, train=train_opom_mask)
    #run(cfgs, mode='train', data=data, train=train_protego_mask_lpips)
    #run(cfgs, mode='train', data=data, train=train_chameleon_mask)
    run(cfgs, mode='eval', data=data)
    """
    exp_name_prefix = cfgs.exp_name
    train_fr_combinations = list(itertools.combinations(cfgs.train_fr_names, 4))
    #combs_to_use = train_fr_combinations
    #combs_to_use = random.Random(cfgs.global_random_seed).sample(train_fr_combinations, k=min(20, len(train_fr_combinations)))
    combs_to_use = train_fr_combinations[4*len(train_fr_combinations)//5:]
    for comb in combs_to_use:
        cfgs.train_fr_names = list(comb)
        cfgs.exp_name = exp_name_prefix + '_' + '-'.join(cfgs.train_fr_names)
        cfgs.mask_name = [f'len4ensemble/{cfgs.exp_name.replace("-", "_")}', 'univ_mask.npy']
        #run(cfgs, mode='train', data=data, train=train_protego_mask)
        run(cfgs, mode='eval', data=data)
    """
