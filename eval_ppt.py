import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" 
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="requests")
import argparse
import sys
import math

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
from omegaconf import OmegaConf

from protego.utils import get_usable_img_paths
from protego.run_exp import run
from protego.FacialRecognition import BASIC_POOL, SPECIAL_POOL
from protego import BASE_PATH

if __name__ == "__main__":
    ####################################################################################################################
    # Configuration
    ####################################################################################################################
    args = argparse.ArgumentParser()
    args.add_argument('--mask_name', type=str, default='default', help='The name of the mask to evaluate.')
    args.add_argument('--exp_name', type=str, default='compression_robustness', help='The name of the evaluation experiment.')
    args.add_argument('--device', type=str, default='cuda:0', help='The device to use for evaluation. (cpu, mps, cuda:0, etc.)')
    args = args.parse_args()

    configs = {
        # Running env
        'global_random_seed': 42, # The global random seed for reproducibility.
        'device': 'cuda:0', # The device to use for evaluation. (cpu, mps, cuda:0, etc.). Will be overwritten by command line argument if provided.
        'exp_name': 'default', # The name of the experiment. Will be overwritten by command line argument if provided.
        
        # Evaluation data configs
        'train_portion': 0.6, # Use 60% of images for training, and the rest for evaluation.
        'shuffle': False, # Whether to shuffle the data before splitting into training and eval sets.
        'uv_gen_align_ldmk': False, # Whether to align the face landmarks before UV map generation. False is faster and, in most cases, sufficient.
        'uv_gen_batch': 8, # The batch size for UV map generation.
        'need_cropping': False, # Whether to crop the face images before training. 
        'fd_name': 'mtcnn', # The face detector to use for cropping. Options: 'mtcnn', 'mobilenet_retinaface_widerface', 'resnet50_retinaface_widerface'
        'crop_loosen': 1., # The looseness factor for face cropping. Greater than 1.0 means more context around the face.
        
        # Mask configs
        'three_d': True, # Whether this is a 3D mask evaluation.
        'epsilon' : 16 / 255., # The maximum perturbation
        'mask_size' : 224, # The size of the mask to train.
        'bin_mask': True, # Whether to use binary mask. If True, the perturbation will be restricted to the face area. 

        # Default eval configs
        'eval_scene': 0,# 0: When evaluating on a protectee, all other protectees' images are outside the attacker's database. 
                        # 1: When evaluating on a protectee, the attacker has access to all other protectees' evaluation images.
        'mask_name': ['default', 'univ_mask.npy'], # The path to the mask to evaluate (experiments/mask_name[0]/{protectee_name}/mask_name[1]). If training, this is ignored. 
        'eval_db': 'face_scrub', # The noise database to use for evaluation. Options: 'face_scrub'
        'eval_fr_names': ['ir50_adaface_casia'] + SPECIAL_POOL, # The FR models to use for evaluation. Choose from BASIC_POOL and SPECIAL_POOL in protego/FacialRecognition.py
        'save_univ_mask': True, # Whether to save the universal mask after training.
        'visualize_interval': 10, # Visualize training and evaluation results every N images.
        'query_portion': 0.5, # The portion of images to use as queries during evaluation.
        'vis_eval': True, # Whether to evaluate visual quality during evaluation.
        'lpips_backbone': "vgg", # The backbone for LPIPS calculation. Options: 'vgg', 'alex'

        # End-to-end eval configs
        'end2end_eval': False, # Whether to perform end-to-end evaluation (uncropped images -> apply mask -> detection -> compression -> FR)
        'strict_crop': True, # Whether to use strict cropping during end2end eval (crop according to the ground-truth face bounding box)
        'smoothing': 'none', # The smoothing method to apply to the mask during end2end eval. Options: 'none', 'gaussian', 'median', 'bilateral'
        'resize_face': True, # Whether to resize the face during end2end eval.
        'jpeg': False, # Whether to apply JPEG compression during end2end eval.

        # Robustness eval configs
        'eval_compression': True, # Whether to evaluate the compression of the mask.
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
        }, 

        # Training hyper-parameters. (Ignored during eval)
        'epoch_num': 100, # Number of training epochs
        'batch_size' : 4, # The batch size for training
        'min_ssim' : 0.95, # The minimum SSIM constraint
        'learning_rate' : 0.01 * (16 / 255.), # Learning rate for optimizer.
        'mask_random_seed': 114, # The random seed for mask initialization.
        'train_fr_names': [n for n in BASIC_POOL if n != 'ir50_adaface_casia'],  # The FR models to use for training. Choose from BASIC_POOL and SPECIAL_POOL in protego/FacialRecognition.py
    }
    ####################################################################################################################
    # Run
    ####################################################################################################################
    cfgs = OmegaConf.create(configs)
    cfgs.mask_name[0] = args.mask_name if '--mask_name' in sys.argv else cfgs.mask_name[0]
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
    with torch.no_grad():
        run(cfgs, mode='eval', data=data)
