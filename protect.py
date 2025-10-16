import os
from typing import List, Dict
import datetime

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import yaml
from omegaconf import OmegaConf

from protego.FacialRecognition import FR
from protego.FaceDetection import FD
from protego.utils import load_imgs, load_mask, build_facedb, build_compressed_face_db, crop_face, complete_del, visualize_mask, eval_masks, compression_eval
from protego import BASE_PATH
from protego.UVMapping import UVGenerator

if __name__ == "__main__":
    ####################################################################################################################
    # Configuration
    ####################################################################################################################
    with torch.no_grad():
        device = torch.device('cuda:0')
        orig_imgs_base_path = os.path.join(BASE_PATH, 'face_db', 'face_scrub')
        mask_base_path = os.path.join(BASE_PATH, 'experiments')
        mask_name = ['default', 'frpair0_mask0_univ_mask.npy']
        orig_imgs_save_path = os.path.join(BASE_PATH, 'results', 'imgs', 'orig')
        prot_imgs_save_path = os.path.join(BASE_PATH, 'results', 'imgs', 'prot')

        smirk_base_path = os.path.join(BASE_PATH, 'smirk')
        smirk_weight_path = os.path.join(smirk_base_path, 'pretrained_models/SMIRK_em1.pt')
        mp_lmk_model_path = os.path.join(smirk_base_path, 'assets/face_landmarker.task')
        uvmapper = UVGenerator(smirk_ckpts_path=smirk_weight_path, smirk_base_path=smirk_base_path, mp_ldmk_model_path=mp_lmk_model_path, device=device)

        protectees = sorted([name for name in os.listdir(orig_imgs_base_path) if not name.startswith(('.', '_'))])
        for protectee in protectees:
            """protectee_orig_save_path = os.path.join(orig_imgs_save_path, protectee)
            protectee_prot_save_path = os.path.join(prot_imgs_save_path, protectee)
            os.makedirs(protectee_orig_save_path, exist_ok=True)
            os.makedirs(protectee_prot_save_path, exist_ok=True)"""
            protectee_orig_imgs = [os.path.join(orig_imgs_base_path, protectee, img_name) for img_name in os.listdir(os.path.join(orig_imgs_base_path, protectee)) if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and not img_name.startswith(('.', '_'))]
            protectee_mask = os.path.join(mask_base_path, mask_name[0], protectee, mask_name[1])
            orig_imgs = load_imgs(img_paths=protectee_orig_imgs, img_sz=224, usage_portion=1.0, drange=1, device=device)
            mask = load_mask(mask_path=protectee_mask, device=device)[[0]]
            uvs, bin_masks, _ = uvmapper.forward(orig_imgs, align_ldmks=False, batch_size=16)
            print(uvs.shape, bin_masks.shape, mask.shape)
            perts = F.grid_sample(mask.repeat(orig_imgs.shape[0], 1, 1, 1), uvs, align_corners=True, mode='bilinear')
            perts *= bin_masks
            prot_imgs = torch.clamp(orig_imgs + perts, 0, 1)
            orig_imgs = orig_imgs.mul(255.).permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]
            prot_imgs = prot_imgs.mul(255.).permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]
            #np.save(os.path.join(protectee_orig_save_path, 'orig_imgs.npy'), orig_imgs)
            #np.save(os.path.join(protectee_prot_save_path, 'prot_imgs.npy'), prot_imgs)
            np.save(os.path.join(orig_imgs_save_path, f'{protectee}.npy'), orig_imgs)
            np.save(os.path.join(prot_imgs_save_path, f'{protectee}.npy'), prot_imgs)
            #perts = (((perts - perts.min()) / (perts.max() - perts.min())) * 255.).permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]
            """for i in range(orig_imgs.shape[0]):
                if i % 20 != 0:
                    continue
                _orig_img = orig_imgs[i].astype(np.uint8)
                _prot_img = prot_imgs[i].astype(np.uint8)
                _pert = perts[i].astype(np.uint8)
                frame = np.hstack([_orig_img, _prot_img, _pert])
                cv2.imwrite(os.path.join(protectee_orig_save_path, f'orig_{i:03d}.png'), frame)"""
