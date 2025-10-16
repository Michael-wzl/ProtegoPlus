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

from protego.protego_train_robust import train_protego_mask_robust
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
        'learning_rate' : 0.01 * (16 / 255.), # 0.01 * (16 / 255.)
        'mask_size' : 224, 
        'mask_random_seed': 114, 
        'bin_mask': True, # Whether to use binary mask. If True, the perturbation will be restricted to the face area. 
        'train_fr_names': [n for n in BASIC_POOL if n != 'ir50_adaface_casia'],

        # Eval configs
        'mask_name': ['default', 'frpair0_mask0_univ_mask.npy'], 
        'eval_db': 'face_scrub',
        'eval_fr_names': ['ir50_adaface_casia'],
        'save_univ_mask': True, 
        'visualize_interval': 10,
        'query_portion': 0.5,
        'vis_eval': True, 
        'lpips_backbone': "vgg", 
        'end2end_eval': True, 
        'resize_face': True, 
        'jpeg': False, 
        'eval_compression': False, # Whether to evaluate the compression of the mask.
        'eval_compression_methods': ['none', 'gaussian', 'median', 'jpeg', 'resize', 'quantize', 'vid_codec'], # The compression methods to evaluate.
        'compression_cfgs' : {
            'none': {}, 
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
                'resz_percentage': 0.4,
                'mode': 'bicubic'
            }, 
            'quantize': {
                'precision': 'uint8'
            }, 
            'vid_codec': {
                'codec': 'h264',
                'crf': 32,
                'preset': 'faster'
            }
        },
        'train_compression_method': ['gaussian', 'median', 'jpeg', 'quantize', 'resize'],
        'train_compression_cfgs' : {
            # Gaussian Filter
            'gaussian': {
                'kernel_size': 9, 
                'sigma': 2.0
            }, 
            # Median Filter
            'median': {
                'kernel_size': 9
            }, 
            'resize': {
                'resz_percentage': 0.4,
                'mode': 'bicubic'
            },
            # JPEG Compression
            'jpeg': {
                'quality': 70
            }, 
            # Quantize
            'quantize': {
                'precision': 'uint8',
                'diff_method': 'ste'
            }
        }
    }
    ####################################################################################################################
    # Run
    ####################################################################################################################
    for method, kwargs in configs['train_compression_cfgs'].items():
        kwargs['differentiable'] = True
    cfgs = OmegaConf.create(configs)
    cfgs.exp_name = args.exp_name if '--exp_name' in sys.argv else cfgs.exp_name
    cfgs.device = args.device if '--device' in sys.argv else cfgs.device
    torch.manual_seed(cfgs.global_random_seed)

    train_portion = 0.6
    shuffle_data = False
    train_data_path = os.path.join(BASE_PATH, 'face_db', 'face_scrub')
    protectees = sorted([name for name in os.listdir(train_data_path) if not name.startswith(('.', '_'))])
    data = {}
    for protectee in protectees:
        protectee_path = os.path.join(train_data_path, protectee)
        imgs = [os.path.join(protectee_path, img_name) for img_name in os.listdir(protectee_path) if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and not img_name.startswith(('.', '_'))]
        train_num = math.floor(len(imgs) * train_portion)
        eval_num = len(imgs) - train_num
        if shuffle_data:
            rand_gen = torch.Generator()
            rand_gen.manual_seed(cfgs.global_random_seed)
            indices = torch.randperm(len(imgs), generator=rand_gen).tolist()
            imgs = [imgs[i] for i in indices]
        data[protectee] = {'train': imgs[:train_num], 'eval': imgs[train_num:]}
    #run(cfgs, mode='train', data=data, train=train_protego_mask_robust)
    run(cfgs, mode='eval', data=data)