import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" 
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="requests")
import datetime
import argparse

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
import torch.nn.functional as F
import cv2
import yaml
import tqdm

from protego.FaceDetection import FD
from protego.utils import load_mask, crop_face, load_imgs
from protego import BASE_PATH
from protego.UVMapping import UVGenerator

if __name__ == "__main__":
    ####################################################################################################################
    # Configuration
    ####################################################################################################################
    args = argparse.ArgumentParser()
    args.add_argument('--mask_name', type=str, default='default', help='The name of the mask to apply.')
    args.add_argument('--protectee', type=str, default='Hugh Grant', help='The name of the protectee whose video to protect.')
    args.add_argument('--device', type=str, default='cuda:0', help='The device to use for protection. (cpu, mps, cuda:0, etc.)')
    args = args.parse_args()
    ####################################################################################################################
    # Run
    ####################################################################################################################
    with torch.no_grad():
        device = torch.device(args.device)
        protectee_name = args.protectee
        mask_names = [args.mask_name, 'univ_mask.npy']

        # Config paths
        src_imgs_base_path = os.path.join(BASE_PATH, 'face_db', 'imgs', protectee_name)
        dst_imgs_base_path = os.path.join(BASE_PATH, 'results', 'imgs', protectee_name)
        mask_path = os.path.join(BASE_PATH, 'experiments', mask_names[0], protectee_name, mask_names[1])
        mask_cfg_path = os.path.join(BASE_PATH, 'experiments', mask_names[0], protectee_name, 'cfgs.yaml')
        with open(mask_cfg_path, 'r') as f:
            mask_cfgs = yaml.safe_load(f)
        three_d = mask_cfgs.get('three_d', True)
        use_bin_mask = mask_cfgs.get('bin_mask', True)
        epsilon = mask_cfgs.get('epsilon', 16 / 255.)
        epsilon = int(float(epsilon) * 255) / 255.
        os.makedirs(dst_imgs_base_path, exist_ok=True)
        smirk_base_path = os.path.join(BASE_PATH, 'smirk')
        smirk_weight_path = os.path.join(smirk_base_path, 'pretrained_models/SMIRK_em1.pt')
        mp_lmk_model_path = os.path.join(smirk_base_path, 'assets/face_landmarker.task')

        # Init models
        fd = FD(model_name='mtcnn', device=device)
        uvmapper = UVGenerator(smirk_ckpts_path=smirk_weight_path, smirk_base_path=smirk_base_path, mp_ldmk_model_path=mp_lmk_model_path, device=device)
        mask = load_mask(mask_path, device=device)

        # Protect images
        imgs_names = [os.path.join(src_imgs_base_path, f) for f in os.listdir(src_imgs_base_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        imgs, valid_img_names = load_imgs(img_paths=imgs_names, img_sz=-1, drange=1, device=device, return_img_paths=True)
        print(f"Loaded {len(imgs)}/{len(imgs_names)} valid images for protection.")
        for img_name in imgs_names:
            if img_name not in valid_img_names:
                print(f"Warning: Unable to load image {img_name}. Skipped.")
        pbar = tqdm.tqdm(enumerate(imgs), total=len(imgs), desc="Protecting images")
        for img_idx, img in pbar:
            img = img.squeeze(0)
            cropped_face, pos = crop_face(img=img, detector=fd, crop_loosen=1., verbose=False)
            if cropped_face is None or pos is None:
                print(f"Warning: No face detected in image {valid_img_names[img_idx]}. Skipping.")
                continue
            face_pos = list(pos)
            orig_face_h, orig_face_w = cropped_face.shape[1], cropped_face.shape[2]
            resized_face = F.interpolate(cropped_face.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False) # [1, 3, 224, 224], 0-1
            uv, bin_mask, _ = uvmapper.forward(imgs=resized_face, align_ldmks=False, batch_size=-1)
            if three_d:
                pert = F.grid_sample(mask, uv, align_corners=True, mode='bilinear')  # [1, 3, 224, 224]
            else:
                pert = mask
            if use_bin_mask:
                pert *= bin_mask
            pert = F.interpolate(pert, size=(orig_face_h, orig_face_w), mode='bilinear', align_corners=False).clamp_(-epsilon, epsilon)
            protected_face: torch.Tensor = (cropped_face + pert.squeeze(0)).clamp_(0., 1.)
            img[:, face_pos[1]:face_pos[3], face_pos[0]:face_pos[2]] = protected_face.contiguous()
            img = img.permute(1, 2, 0).mul(255.).to(torch.uint8)[:, :, [2, 1, 0]].cpu().contiguous().numpy()
            dst_img_path = os.path.join(dst_imgs_base_path, "protected_" + os.path.basename(valid_img_names[img_idx]))
            cv2.imwrite(dst_img_path, img)
        print(f"Finished protecting images. Protected images are saved to {dst_imgs_base_path}.")
