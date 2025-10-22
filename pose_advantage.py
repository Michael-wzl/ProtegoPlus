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

from protego.protego_train import train_protego_mask
from protego.protego_train_lpips import train_protego_mask_lpips
from protego.chameleon_train import train_chameleon_mask
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
        'need_cropping': True, 
        'fd_name': 'mtcnn', 
        'crop_loosen': 0.9, 
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
        'mask_name': ['all_sides', 'univ_mask.npy'], 
        'eval_db': 'face_scrub',
        'eval_fr_names': ['ir50_adaface_casia'],
        'save_univ_mask': True, 
        'visualize_interval': 10,
        'query_portion': 0.5,
        'vis_eval': True, 
        'lpips_backbone': "vgg", 
        'end2end_eval': False, 
        'strict_crop': True, 
        'resize_face': True, 
        'jpeg': False,
        'smoothing': 'none', # Options: 'gaussian', 'median', 'none'
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

    train_portion = 0.6
    shuffle_data = False
    front_data_path = os.path.join(BASE_PATH, 'face_db', 'BC_front')
    left_data_path = os.path.join(BASE_PATH, 'face_db', 'BC_left')
    right_data_path = os.path.join(BASE_PATH, 'face_db', 'BC_right')
    protectees = ["Bradley_Cooper"]
    data = {}
    for protectee in protectees:
        front_imgs = [os.path.join(front_data_path, img_name) for img_name in os.listdir(front_data_path) if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and not img_name.startswith(('.', '_'))]
        left_imgs = [os.path.join(left_data_path, img_name) for img_name in os.listdir(left_data_path) if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and not img_name.startswith(('.', '_'))]
        right_imgs = [os.path.join(right_data_path, img_name) for img_name in os.listdir(right_data_path) if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and not img_name.startswith(('.', '_'))]
        
        front_train_num = math.floor(len(front_imgs) * train_portion)
        front_eval_num = len(front_imgs) - front_train_num
        left_train_num = math.floor(len(left_imgs) * train_portion)
        left_eval_num = len(left_imgs) - left_train_num
        right_train_num = math.floor(len(right_imgs) * train_portion)
        right_eval_num = len(right_imgs) - right_train_num
        """data[protectee] = {'train': front_imgs[:front_train_num], 
                           'eval': front_imgs[front_train_num:]}"""
        """data[protectee] = {'train': left_imgs[:left_train_num], 
                           'eval': left_imgs[left_train_num:]}"""
        """data[protectee] = {'train': right_imgs[:right_train_num], 
                           'eval': right_imgs[right_train_num:]}"""
        data[protectee] = {'train': right_imgs[:right_train_num] + left_imgs[:left_train_num] + front_imgs[:front_train_num],
                           'eval': right_imgs[right_train_num:] + left_imgs[left_train_num:] + front_imgs[front_train_num:]}
        #data[protectee] = {'eval': right_imgs[right_train_num:] + left_imgs[left_train_num:] + front_imgs[front_train_num:]}
    #run(cfgs, mode='train', data=data, train=train_protego_mask)
    run(cfgs, mode='eval', data=data)
