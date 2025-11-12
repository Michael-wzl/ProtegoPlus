import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="requests")
from typing import List, Dict, Any

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
import torch.nn.functional as F
import numpy as np
import yaml
import tqdm

from protego.FacialRecognition import FR
from protego.utils import load_mask, build_facedb, load_imgs, kmeans, visualize_mask, get_usable_img_paths
from protego import BASE_PATH
from protego.UVMapping import UVGenerator

def cal_purity(preds: np.ndarray, img_paths: List[str]) -> Dict[str, Any]:
    res_by_label, res_by_cluster = {}, {}
    for img_idx in range(len(img_paths)):
        cluster_id = int(preds[img_idx])
        img_path = img_paths[img_idx]
        label = img_path.split('/')[-2]
        if cluster_id not in res_by_cluster:
            res_by_cluster[cluster_id] = []
        res_by_cluster[cluster_id].append(img_path)
        if label not in res_by_label:
            res_by_label[label] = {}
            res_by_label[label]['img_num'] = 0
            res_by_label[label]['labels'] = {}
        if cluster_id not in res_by_label[label]['labels']:
            res_by_label[label]['labels'][cluster_id] = 0
        res_by_label[label]['img_num'] += 1
        res_by_label[label]['labels'][cluster_id] += 1
    purities = []
    for label, v in res_by_label.items():
        max_cnt = max(v['labels'].values())
        purity = max_cnt / v['img_num']
        purities.append(purity)
    return {
        'res_by_label': res_by_label,
        'res_by_cluster': res_by_cluster,
        'purities': purities,
        'avg_purity': np.mean(purities)
    }


if __name__ == "__main__":
    with torch.no_grad():
        ####################################################################################################################
        # Configuration
        ####################################################################################################################
        device = torch.device('cuda:7')
        face_db_name = 'face_scrub'
        mask_name = ['default_opom', 'univ_mask.npy']
        epsilon = 16 / 255.
        three_d = False
        bin_mask = False
        fr_name = "ir50_adaface_casia"
        prot_portion = 0.2 # The portion of images to be protected for each protectee.
        res_save_name = f"{face_db_name}_{mask_name[0]}_{fr_name}_prot{int(prot_portion*100)}.yaml"
        sanity_check = True
        ####################################################################################################################
        res_save_base_path = os.path.join(BASE_PATH, 'results', 'cluster')
        face_db_base = os.path.join(BASE_PATH, 'face_db', face_db_name)
        noise_db_path = os.path.join(face_db_base, '_noise_db')
        protectee_base_path = face_db_base
        mask_base_path = os.path.join(BASE_PATH, 'experiments', mask_name[0])

        fr = FR(model_name=fr_name, device=device)
        if prot_portion > 0.:
            smirk_base_path = os.path.join(BASE_PATH, 'smirk')
            smirk_weight_path = os.path.join(smirk_base_path, 'pretrained_models/SMIRK_em1.pt')
            mp_lmk_model_path = os.path.join(smirk_base_path, 'assets/face_landmarker.task')
            uvmapper = UVGenerator(smirk_ckpts_path=smirk_weight_path, smirk_base_path=smirk_base_path, mp_ldmk_model_path=mp_lmk_model_path, device=device)
        
        num_ids = 0
        noise_img_names, noise_features = [], []
        # Load noise images
        noise_db = build_facedb(db_path=noise_db_path, fr=fr, device=device, return_img_paths=True)
        pbar = tqdm.tqdm(noise_db.items())
        for name, (features, img_names) in pbar:
            pbar.set_description(f"Processing noise {name}")
            num_ids += 1
            noise_features.append(features)
            noise_img_names.extend(img_names)
        noise_features = torch.cat(noise_features, dim=0)  # (N, D)
        print(f"Loaded {noise_features.shape[0]} noise images from {num_ids} noise IDs.")
        
        # Load protectee images
        protectee_orig_img_names, protectee_prot_img_names, protectee_orig_features, protectee_prot_features = [], [], [], []
        pbar = tqdm.tqdm(os.listdir(protectee_base_path))
        for name in pbar:
            pbar.set_description(f"Processing protectee {name}")
            personal_path = os.path.join(protectee_base_path, name)
            if name.startswith(('.', '_')):
                print(f"Skip {personal_path}.")
                continue
            mask = load_mask(os.path.join(mask_base_path, name, mask_name[1]), device=device)
            img_names = get_usable_img_paths(personal_path)
            orig_imgs = load_imgs(img_paths=img_names, img_sz=224, usage_portion=1., drange=1, device=device, return_img_paths=False)
            num_orig = orig_imgs.shape[0]
            num_prot = int(prot_portion * num_orig)
            num_orig -= num_prot
            if prot_portion > 0.:
                prot_imgs = orig_imgs[:num_prot]
                uvs, bin_masks, _ = uvmapper.forward(prot_imgs, align_ldmks=False, batch_size=16)
                if three_d:
                    perts = torch.clamp(F.grid_sample(mask.repeat(prot_imgs.shape[0], 1, 1, 1), uvs, align_corners=True, mode='bilinear'), -epsilon, epsilon)
                else:
                    perts = torch.clamp(mask.repeat(prot_imgs.shape[0], 1, 1, 1), -epsilon, epsilon)
                if bin_mask:
                    perts *= bin_masks
                prot_imgs = torch.clamp(prot_imgs + perts, 0., 1.)
                prot_features = fr(prot_imgs)
                prot_img_names = [img_name+'<prot>' for img_name in img_names[:num_prot]]
                protectee_prot_img_names.extend(prot_img_names)
                protectee_prot_features.append(prot_features)
                if sanity_check:
                    for i in range(len(prot_imgs)):
                        if i % 20 != 0:
                            continue
                        visualize_mask(orig_img=orig_imgs[i], 
                                       uv=uvs[i],
                                       bin_mask=bin_masks[i],
                                       univ_mask=mask, 
                                       save_path=os.path.join(res_save_base_path, f'sanity_check_{name}_{i}.jpg'),
                                       epsilon=epsilon, 
                                       use_bin_mask=bin_mask, 
                                       three_d=three_d)
            num_ids += 1
            if num_orig <= 0:
                continue
            orig_imgs = orig_imgs[num_prot:]
            orig_features = fr(orig_imgs)
            orig_img_names = img_names[num_prot:]
            protectee_orig_img_names.extend(orig_img_names)
            protectee_orig_features.append(orig_features)
        if len(protectee_orig_features) > 0:
            protectee_orig_features = torch.cat(protectee_orig_features, dim=0)
        if prot_portion > 0.:
            protectee_prot_features = torch.cat(protectee_prot_features, dim=0)
        print(f"Loaded {len(protectee_orig_features)} original protectee images and {len(protectee_prot_features)} protected protectee images.")
        
        img_paths = noise_img_names + protectee_orig_img_names + protectee_prot_img_names
        if len(protectee_orig_img_names) > 0 and len(protectee_prot_img_names) > 0:
            features: torch.Tensor = torch.cat([noise_features, protectee_orig_features, protectee_prot_features], dim=0)  # (N, D)
        elif len(protectee_orig_img_names) > 0 and len(protectee_prot_img_names) == 0:
            features: torch.Tensor = torch.cat([noise_features, protectee_orig_features], dim=0)  # (N, D)
        elif len(protectee_orig_img_names) == 0 and len(protectee_prot_img_names) > 0:
            features: torch.Tensor = torch.cat([noise_features, protectee_prot_features], dim=0)  # (N, D)
        else:
            raise ValueError("No protectee images loaded.")
        assert len(img_paths) == features.shape[0]
        orig_noise_num = len(noise_img_names) + len(protectee_orig_img_names)
        preds = kmeans(features=features.to(device), n_clusters=num_ids, rand_seed=42).cpu().numpy()
        assert len(preds) == len(img_paths)

        orig_noise_preds = preds[:orig_noise_num]
        prot_preds = preds[orig_noise_num:]
        overall_res = {}
        res = cal_purity(orig_noise_preds, img_paths[:orig_noise_num])
        overall_res.update(res['res_by_cluster'])
        for label, v in res['res_by_label'].items():
            max_cnt = max(v['labels'].values())
            purity = max_cnt / v['img_num']
            print(f"{label}: purity={purity:.4f} ({max_cnt}/{v['img_num']})")
        print(f"Average purity for original and noise images: {res['avg_purity']:.4f}")
        if prot_portion > 0.:
            res = cal_purity(prot_preds, img_paths[orig_noise_num:])
            overall_res.update(res['res_by_cluster'])
            for label, v in res['res_by_label'].items():
                max_cnt = max(v['labels'].values())
                purity = max_cnt / v['img_num']
                print(f"{label}: purity={purity:.4f} ({max_cnt}/{v['img_num']})")
            print(f"Average purity for protected images: {res['avg_purity']:.4f}")
        with open(os.path.join(res_save_base_path, res_save_name), 'w') as f:
            yaml.dump(overall_res, f)