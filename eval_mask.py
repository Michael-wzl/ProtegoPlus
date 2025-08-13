import os
from typing import List, Dict, Any
import sys

import argparse
from omegaconf import OmegaConf
import numpy as np
import torch
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
from torch.utils.data import DataLoader

from FacialRecognition import *
from run import run1, run2

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

def load_mask(cfgs: OmegaConf, 
              protectee: str) -> Dict[str, Any]:
    mask = np.load(os.path.join(BASE_PATH, 'experiments', cfgs.mask_name[0], protectee, cfgs.mask_name[1]), allow_pickle=True)[[0]]
    return {'univ':mask, 'train': None, 'test': None} 

if __name__ == "__main__":
    ####################################################################################################################
    # Configuration
    ####################################################################################################################
    # This code is adopted from the training code of Protego (which is currently not provided), therefore some configurations are not used in the evaluation.
    configs = {
        # Useful configurations
        'device': 'cuda:0',  
        'exp_name': 'eval_default',  # The name of the subfolder under experiments/ where the evaluation results and visualizations will be saved.
        'protectees' : 'all',  # 'all', a list of protectees to include or exclude(add a '!' to the front of the list.), or the first(n>0)/last(n<0) n-1 protectees in the dataset
        'usage_portion' : 1.0, # The portion of the dataset visible to the model. 1.0 means all images are visible.
        'train_portion' : 0.6,  # The portion of the dataset to use for training. The rest will be used for testing.
        'query_portion': 0.5, # The portion of the test set to use for querying. The rest will be used as gallery.
        'random_split' : False,  # Whether to randomly split the dataset into train and test sets.
        'shuffle' : False, # Whether to shuffle the dataset before splitting.
        'bin_mask': True, # Whether to use binary mask. If True, the perturbation will be restricted to the face area. 
        'eval_scene': 1, # 0: When evaluating one user, the rest of the selected users will not be included in the noise db. 1: When evaluating one user, the rest of the selected users will be included in the noise db.
        'vis_eval': True, # Whether to evaluate the visual quality of the mask. 
        'compression_eval': False, # Whether to evaluate the compression of the mask.
        'visualize_interval': 1, # Visualize the mask every n test images. Set it to any value <= 0 to disable saving additional masks.
        'eval_fr_names': ['ir50_adaface_casia'], # The names of the facial recognition models to use for evaluation.
        'mask_name': ['default', 'frpair0_mask0_univ_mask.npy'], # Only the first element need to be changed. The first element is the name of the folder in 'experiments' of which the mask is loaded from. The second element is the name of the mask file to load. 
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

        # The configurations below are useless for evaluation or should not be changed. But they are required for the code to run.
        'three_d': True, 
        'epoch_num': 100,  
        'batch_size' : 4,  
        'epsilon' : 16 / 255., 
        'min_ssim' : 0.95, 
        'learning_rate' : 0.01 * (16 / 255.), # 0.01 * (16 / 255.)
        'mask_size' : 224, 
        'mask_seeds': [114],  #[114, 514, 191, 98, 10, 42, 35] will also serve as the round of experiments for each fr_pair and each person. 
        'global_random_seed' : 42,  # The global random seed for reproducibility. Random initialization of the mask will use different seeds for each round of experiments as defined in 'mask_seeds'.
        'eval_db' : "face_scrub", # The database to use for evaluation.
        'save_univ_mask': False, # Whether to save the universal mask as npy after training. 
        'save_additional_mask_interval': -1, # Save the additional masks every n test images. Set it to any value <= 0 to disable saving additional masks.
        'fr_pairs' : [(["ir50_adaface_casia"], ["ir50_adaface_casia"])]
    }
    ####################################################################################################################
    # Run
    ####################################################################################################################
    cfgs = OmegaConf.create(configs)
    _fr_pair = list(cfgs.fr_pairs[0])
    _fr_pair[1] = cfgs.eval_fr_names
    cfgs.fr_pairs[0] = tuple(_fr_pair)
    torch.manual_seed(cfgs.global_random_seed)
    if cfgs.eval_scene == 0:
        run1(load_mask, cfgs)
    elif cfgs.eval_scene == 1:
        run2(load_mask, cfgs)

    