import os
import shutil

import torch
import torch.nn.functional as F
import numpy as np
import cv2

from protego.utils import load_imgs, load_mask
from protego.UVMapping import UVGenerator
from protego import BASE_PATH
import yaml

cluster_res = "/home/zlwang/ProtegoPlus/results/cluster/face_scrub_default_frpair0_mask0_univ_mask_ir50_adaface_casia_prot50_nokmeans.yaml"
dst_base_dir = "/home/zlwang/ProtegoPlus/face_db/fs_prot50_nokmeans"
with open(cluster_res, 'r') as f:
    cluster_res = yaml.safe_load(f)
if "prot" not in cluster_res and "prot" not in dst_base_dir:
    for label, img_paths in cluster_res.items():
        print(f'Cluster {label}, size: {len(img_paths)}')
        if len(img_paths) == 0:
            print(f'Cluster {label} is empty !!!')
            break
        label = f"id_{int(label):03d}"
        id_path = os.path.join(dst_base_dir, label)
        os.makedirs(id_path, exist_ok=True)
        for img_path in img_paths:
            new_path = os.path.join(id_path, img_path.split('/')[-1])
            shutil.copy(img_path, new_path)
else:
    with torch.no_grad():
        device = torch.device('cuda:0')
        smirk_base_path = os.path.join(BASE_PATH, 'smirk')
        smirk_weight_path = os.path.join(smirk_base_path, 'pretrained_models/SMIRK_em1.pt')
        mp_lmk_model_path = os.path.join(smirk_base_path, 'assets/face_landmarker.task')
        uvmapper = UVGenerator(smirk_ckpts_path=smirk_weight_path, smirk_base_path=smirk_base_path, mp_ldmk_model_path=mp_lmk_model_path, device=device)
        all_masks = {}
        mask_base_path = os.path.join(BASE_PATH, 'experiments')
        mask_name = ['default', 'frpair0_mask0_univ_mask.npy']
        epsilon = 16 / 255.
        for protectee in os.listdir(os.path.join(mask_base_path, mask_name[0])):
            if protectee.startswith(('.', '_')):
                continue
            mask_path = os.path.join(mask_base_path, mask_name[0], protectee, mask_name[1])
            all_masks[protectee] = load_mask(mask_path=mask_path, device=device)[[0]]
        for label, img_paths in cluster_res.items():
            print(f'Cluster {label}, size: {len(img_paths)}')
            if len(img_paths) == 0:
                print(f'Cluster {label} is empty !!!')
                break
            label = f"id_{int(label):03d}" if label.isdigit() else label
            id_path = os.path.join(dst_base_dir, label)
            os.makedirs(id_path, exist_ok=True)
            need_prot_imgs = []
            no_prot_imgs = []
            for img_path in img_paths:
                if img_path.endswith("/prot"):
                    need_prot_imgs.append("/".join(img_path.split('/')[:-1]))
                else:
                    no_prot_imgs.append(img_path)
            for img_path in no_prot_imgs:
                orig_img = load_imgs(img_paths=[img_path], img_sz=-1, usage_portion=1.0, drange=255, device=torch.device('cpu'))[0].squeeze(0).numpy().astype(np.float32) # [3, H, W], RGB, 255, np.float32
                cv2.imwrite(f"/home/zlwang/ProtegoPlus/trash/tmptmp/{img_path.split('/')[-2]}_orig.png", orig_img.copy().transpose(1, 2, 0)[..., ::-1].astype(np.uint8))
                #print(orig_img.shape)
                new_name = img_path.split('/')[-1].split('.')[0] + ".npy"
                new_path = os.path.join(id_path, new_name)
                np.save(new_path, orig_img)
            for img_path in need_prot_imgs:
                #print(img_path)
                protectee_name = img_path.split('/')[-2]
                protectee_mask = all_masks[protectee_name]
                orig_img = load_imgs(img_paths=[img_path], img_sz=224, usage_portion=1.0, drange=1, device=device)
                #print(type(orig_img), orig_img.shape)
                uv, bin_mask, _ = uvmapper.forward(imgs=orig_img, align_ldmks=False, batch_size=-1)
                pert = torch.clamp(F.grid_sample(protectee_mask.repeat(orig_img.shape[0], 1, 1, 1), uv, align_corners=True, mode='bilinear') * bin_mask, -epsilon, epsilon)
                prot_img = torch.clamp(orig_img + pert, 0, 1)
                prot_img = prot_img.mul(255.).squeeze(0).contiguous().cpu().numpy().astype(np.float32) # [3, H, W], RGB, 255, np.float32
                cv2.imwrite(f"/home/zlwang/ProtegoPlus/trash/tmptmp/{protectee_name}_prot.png", prot_img.copy().transpose(1, 2, 0)[..., ::-1].astype(np.uint8))
                #print(prot_img.shape)
                new_name = img_path.split('/')[-1].split('.')[0] + ".npy"
                new_path = os.path.join(id_path, new_name)
                np.save(new_path, prot_img)