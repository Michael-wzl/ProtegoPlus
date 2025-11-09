import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="requests")
from typing import List, Dict

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
import torch.nn.functional as F
import numpy as np
import yaml
import cv2

from protego.utils import load_imgs, load_mask
from protego.UVMapping import UVGenerator
from protego import BASE_PATH

def save_npy_imgs(imgs: torch.Tensor, save_base_path: str, img_names: List[str]):
    imgs = imgs.mul(255.).detach().cpu().contiguous().numpy() # [N, C, H, W], 0-255
    for img, img_name in zip(imgs, img_names):
        save_path = os.path.join(save_base_path, os.path.basename(img_name).split('.')[0]+'.npy')
        np.save(save_path, img) # Save as numpy array with shape [C, H, W], range 0-255

def visualize_npy_imgs(npy_path: str, save_path: str):
    img = np.load(npy_path) # [C, H, W], 0-255
    img = np.transpose(img, (1, 2, 0)).astype(np.uint8) # [H, W, C], 0-255
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    with torch.no_grad():
        ####################################################################################################################
        # Configuration
        ####################################################################################################################
        device = torch.device('cuda:0')
        cluster_res_name = "face_scrub_default_cham_ir50_adaface_casia_prot20.yaml"
        three_d = False
        bin_mask = False
        epsilon = 16 / 255.
        mask_name = ['default_cham', 'univ_mask.npy']
        save_path = "/home/zlwang/AdaFace/data/fs_prot20_cham"
        sanity_check = True
        ####################################################################################################################
        smirk_base_path = os.path.join(BASE_PATH, 'smirk')
        smirk_weight_path = os.path.join(smirk_base_path, 'pretrained_models/SMIRK_em1.pt')
        mp_lmk_model_path = os.path.join(smirk_base_path, 'assets/face_landmarker.task')
        uvmapper = UVGenerator(smirk_ckpts_path=smirk_weight_path, smirk_base_path=smirk_base_path, mp_ldmk_model_path=mp_lmk_model_path, device=device)
        
        all_masks: Dict[str, torch.Tensor] = {}
        with open(os.path.join(BASE_PATH, 'results', 'cluster', cluster_res_name), 'r') as f:
            cluster_res = yaml.safe_load(f)
        for cluster_id, img_paths in cluster_res.items():
            cluster_id = int(cluster_id)
            save_path_id = os.path.join(save_path, f"cluster_{cluster_id:03d}")
            os.makedirs(save_path_id, exist_ok=True)
            prot_idxs, orig_idxs, img_names, masks = [], [], [], []
            for img_idx, img_path in enumerate(img_paths):
                label = img_path.split('/')[-2]
                if img_path.endswith('<prot>'):
                    prot_idxs.append(img_idx)
                    img_path = img_path[:-len('<prot>')]
                    if label not in all_masks:
                        mask = load_mask(os.path.join(BASE_PATH, 'experiments', mask_name[0], label, mask_name[1]), device=device)
                        all_masks[label] = mask
                    else:
                        mask = all_masks[label]
                else:
                    mask = None
                    orig_idxs.append(img_idx)
                img_names.append(img_path)
                masks.append(mask)
            imgs: torch.Tensor = load_imgs(img_paths=img_names, img_sz=224, usage_portion=1., drange=1, device=device)
            if len(orig_idxs) > 0:
                orig_imgs = imgs[orig_idxs]
                orig_img_names = [img_names[i] for i in orig_idxs]
                save_npy_imgs(orig_imgs, save_path_id, orig_img_names)
                if sanity_check and cluster_id % 75 == 0:
                    npy_path = os.path.join(save_path_id, os.path.basename(orig_img_names[0]).split('.')[0]+'.npy')
                    vis_save_path = os.path.join("/home/zlwang/ProtegoPlus/trash/_tmp", 'orig'+os.path.basename(orig_img_names[0]))
                    visualize_npy_imgs(npy_path, vis_save_path)
            if len(prot_idxs) > 0:
                prot_imgs = imgs[prot_idxs]
                prot_img_names = [img_names[i] for i in prot_idxs]
                masks = torch.cat([masks[i] for i in prot_idxs], dim=0)
                uvs, bin_masks, _ = uvmapper.forward(prot_imgs, align_ldmks=False, batch_size=16)
                #print(uvs.shape, bin_masks.shape, masks.shape)
                if three_d:
                    perts = torch.clamp(F.grid_sample(masks, uvs, align_corners=True, mode='bilinear'), -epsilon, epsilon)
                else:
                    perts = torch.clamp(masks, -epsilon, epsilon)
                if bin_mask:
                    perts *= bin_masks
                prot_imgs = torch.clamp(prot_imgs + perts, 0., 1.)
                save_npy_imgs(prot_imgs, save_path_id, prot_img_names)
                if sanity_check:
                    npy_path = os.path.join(save_path_id, os.path.basename(prot_img_names[0]).split('.')[0]+'.npy')
                    vis_save_path = os.path.join("/home/zlwang/ProtegoPlus/trash/_tmp", 'prot'+os.path.basename(prot_img_names[0]))
                    visualize_npy_imgs(npy_path, vis_save_path)
