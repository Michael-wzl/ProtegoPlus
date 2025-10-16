import os

import torch
import torch.nn.functional as F
import cv2
import numpy as np

from protego.compression import compress
from protego.utils import load_imgs, load_mask
from protego import BASE_PATH
from protego.UVMapping import UVGenerator


if __name__ == "__main__":
    with torch.no_grad():
        device = torch.device("cuda:0")
        mask_name = ['default', 'frpair0_mask0_univ_mask.npy']
        epsilon = 16 / 255.
        img_base_path = "/home/zlwang/ProtegoPlus/face_db/face_scrub/Bradley_Cooper"
        eval_compression_methods = ['gaussian', 'median', 'jpeg', 'resize', 'quantize', 'vid_codec']
        compression_cfgs = {
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
        }
        smirk_base_path = os.path.join(BASE_PATH, 'smirk')
        smirk_weight_path = os.path.join(smirk_base_path, 'pretrained_models/SMIRK_em1.pt')
        mp_lmk_model_path = os.path.join(smirk_base_path, 'assets/face_landmarker.task')
        uvmapper = UVGenerator(smirk_ckpts_path=smirk_weight_path, smirk_base_path=smirk_base_path, mp_ldmk_model_path=mp_lmk_model_path, device=device)

        mask = load_mask(mask_path=os.path.join(f"/home/zlwang/ProtegoPlus/experiments/{mask_name[0]}/Bradley_Cooper/{mask_name[1]}"), device=device)
        imgs_path = [os.path.join(img_base_path, fname) for fname in os.listdir(img_base_path) if fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))]
        imgs_path = imgs_path[:20]
        imgs = load_imgs(img_paths=imgs_path, img_sz=224, device=device)
        uvs, bins, _ = uvmapper.forward(imgs)
        prot_imgs = torch.clamp(imgs + torch.clamp(F.grid_sample(mask.repeat(uvs.shape[0], 1, 1, 1), uvs, align_corners=True, mode='bilinear'), -epsilon, epsilon) * bins, 0, 1)
        for method in eval_compression_methods:
            print(f"Applying {method} compression ...")
            prot_imgs_compressed = compress(imgs=prot_imgs, method=method, differentiable=False if method != 'quantize' else True, **compression_cfgs[method])
            orig_imgs_compressed = compress(imgs=imgs, method=method, differentiable=False if method != 'quantize' else True, **compression_cfgs[method])
            if method == 'quantize':
                #print(prot_imgs_compressed.min(), prot_imgs_compressed.max(), orig_imgs_compressed.min(), orig_imgs_compressed.max())
                prot_imgs_compressed = prot_imgs_compressed / 255.
                orig_imgs_compressed = orig_imgs_compressed / 255.
            for i in range(len(imgs_path)):
                _prot = prot_imgs[i].permute(1, 2, 0).mul(255.).cpu().numpy().astype(np.uint8)
                _prot =cv2.cvtColor(_prot, cv2.COLOR_RGB2BGR)
                _prot = cv2.putText(_prot, f"Protected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                _orig = imgs[i].permute(1, 2, 0).mul(255.).cpu().numpy().astype(np.uint8)
                _orig = cv2.cvtColor(_orig, cv2.COLOR_RGB2BGR)
                _orig = cv2.putText(_orig, f"Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                _prot_compressed = prot_imgs_compressed[i].permute(1, 2, 0).mul(255.).cpu().numpy().astype(np.uint8)
                _prot_compressed = cv2.cvtColor(_prot_compressed, cv2.COLOR_RGB2BGR)
                _prot_compressed = cv2.putText(_prot_compressed, f"Protected {method}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                _orig_compressed = orig_imgs_compressed[i].permute(1, 2, 0).mul(255.).cpu().numpy().astype(np.uint8)
                _orig_compressed = cv2.cvtColor(_orig_compressed, cv2.COLOR_RGB2BGR)
                _orig_compressed = cv2.putText(_orig_compressed, f"Original {method}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                _frame = cv2.vconcat([cv2.hconcat([_orig, _orig_compressed]), cv2.hconcat([_prot, _prot_compressed])])
                cv2.imwrite(f"/home/zlwang/ProtegoPlus/trash/compress_vis/comp_{method}_{i}.png", _frame)
