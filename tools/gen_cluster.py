import os

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import tqdm

from protego.FacialRecognition import FR
from protego.utils import load_mask, load_imgs, kmeans, visualize_mask
from protego import BASE_PATH
from protego.UVMapping import UVGenerator

def cluster_features(features, n_clusters):
    return kmeans(features, n_clusters, rand_seed=0)

if __name__ == "__main__":
    ####################################################################################################################
    # Configuration
    ####################################################################################################################
    with torch.no_grad():
        device = torch.device('cuda:0')
        face_db_name = 'face_scrub'
        mask_name = ['default', 'frpair0_mask0_univ_mask.npy']
        epsilon = 16 / 255.
        fr_name = "ir50_adaface_casia"
        prot_portion = 0.2 # The portion of images to be protected for each protectee.
        sanity_check = True

        face_db_base = os.path.join(BASE_PATH, 'face_db', face_db_name)
        noise_db_path = os.path.join(face_db_base, '_noise_db')
        protectee_base_path = face_db_base
        mask_base_path = os.path.join(BASE_PATH, 'experiments', mask_name[0])

        # Init
        fr = FR(model_name=fr_name, device=device)
        if prot_portion > 0:
            smirk_base_path = os.path.join(BASE_PATH, 'smirk')
            smirk_weight_path = os.path.join(smirk_base_path, 'pretrained_models/SMIRK_em1.pt')
            mp_lmk_model_path = os.path.join(smirk_base_path, 'assets/face_landmarker.task')
            uvmapper = UVGenerator(smirk_ckpts_path=smirk_weight_path, smirk_base_path=smirk_base_path, mp_ldmk_model_path=mp_lmk_model_path, device=device)

        persons = 0
        # Load noise images
        noise_img_names, noise_features = [], []
        pbar = tqdm.tqdm(os.listdir(noise_db_path), desc='Loading noise images')
        for name in pbar:
            if name.lower().startswith(('.', '_')):
                continue
            personal_path = os.path.join(noise_db_path, name)
            personal_imgs = []
            for img_name in os.listdir(personal_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and not img_name.startswith(('.', '_')):
                    img_path = os.path.join(personal_path, img_name)
                    personal_imgs.append(os.path.join(personal_path, img_name))
            orig_imgs = load_imgs(img_paths=personal_imgs, img_sz=224, usage_portion=1.0, drange=1, device=device)
            noise_features.append(fr(orig_imgs).cpu())
            noise_img_names.extend(personal_imgs)
            persons += 1
        noise_features = torch.cat(noise_features, dim=0)

        # Load protectee images
        protectee_orig_img_names, protectee_prot_img_names, protectee_orig_features, protectee_prot_features = [], [], [], []
        pbar = tqdm.tqdm(os.listdir(protectee_base_path), desc='Loading protectee images')
        for name in pbar:
            if name.lower().startswith(('.', '_')):
                continue
            personal_path = os.path.join(protectee_base_path, name)
            personal_imgs = [os.path.join(personal_path, fname) for fname in os.listdir(personal_path) if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and not fname.startswith(('.', '_'))]
            orig_imgs = load_imgs(img_paths=personal_imgs, img_sz=224, usage_portion=1.0, drange=1, device=device)
            orig_num = int(len(personal_imgs) * (1 - prot_portion))
            if prot_portion > 0:
                uvs, bin_masks, _ = uvmapper.forward(orig_imgs, align_ldmks=False, batch_size=16)
                protectee_mask = load_mask(mask_path=os.path.join(mask_base_path, name, mask_name[1]), device=device)[[0]]
                perts = torch.clamp(F.grid_sample(protectee_mask.repeat(orig_imgs.shape[0], 1, 1, 1), uvs, align_corners=True, mode='bilinear'), -epsilon, epsilon)* bin_masks
                prot_imgs = torch.clamp(orig_imgs + perts, 0, 1)
                protectee_prot_img_names.extend([n+"/prot" for n in personal_imgs[orig_num:]])
                protectee_prot_features.append(fr(prot_imgs[orig_num:]).cpu())
                if sanity_check:
                    for i in range(len(prot_imgs)):
                        if i % 20 != 0:
                            continue
                        visualize_mask(orig_imgs[i], 
                                       uv=uvs[i], 
                                       bin_mask=bin_masks[i], 
                                       univ_mask=protectee_mask.clone(), 
                                       save_path=os.path.join(BASE_PATH, 'results', 'cluster', f'sanity_check_{name}_{i}.png'), 
                                       epsilon=epsilon, 
                                       use_bin_mask=True, 
                                       three_d=True)
            if orig_num > 0:
                protectee_orig_img_names.extend(personal_imgs[:orig_num])
                protectee_orig_features.append(fr(orig_imgs[:orig_num]).cpu())
            persons += 1
        if len(protectee_orig_features) > 0:
            protectee_orig_features = torch.cat(protectee_orig_features, dim=0)
        if prot_portion > 0:
            protectee_prot_features = torch.cat(protectee_prot_features, dim=0)

        overall_res = {}
        if not prot_portion > 0:
            img_paths = noise_img_names + protectee_orig_img_names
            features = torch.cat([noise_features, protectee_orig_features], dim=0)
            orig_noise_num = len(noise_img_names) + len(protectee_orig_img_names)
        elif prot_portion > 0 and len(protectee_orig_features) == 0:
            img_paths = noise_img_names + protectee_prot_img_names
            features = torch.cat([noise_features, protectee_prot_features], dim=0)
            orig_noise_num = len(noise_img_names)
        elif prot_portion > 0 and len(protectee_orig_features) > 0:
            img_paths = noise_img_names + protectee_orig_img_names + protectee_prot_img_names
            features = torch.cat([noise_features, protectee_orig_features, protectee_prot_features], dim=0)
            orig_noise_num = len(noise_img_names) + len(protectee_orig_img_names)
        else:
            raise ValueError(f"Unknown case of prot_portion {prot_portion} and len(protectee_orig_features) {len(protectee_orig_features)}")
        print(f'Clustering {features.shape[0]} images of {persons} persons...')
        preds = cluster_features(features.to(device), n_clusters=persons).cpu().numpy().tolist()
        assert len(preds) == len(img_paths)
        # We now check
        # 1. The quality of the clustering of original images and noise images
        orig_noise_preds = preds[:orig_noise_num]
        res = {}
        for idx in range(orig_noise_num):
            label = orig_noise_preds[idx]
            if label not in overall_res:
                overall_res[label] = []
            overall_res[label].append(img_paths[idx])
            name = img_paths[idx].split('/')[-2]
            if name not in res:
                res[name] = {}
                res[name]['img_num'] = 0
                res[name]['labels'] = {}
            if label not in res[name]['labels']:
                res[name]['labels'][label] = 0
            res[name]['img_num'] += 1
            res[name]['labels'][label] += 1
        purities = []
        for name, v in res.items():
            max_count = max(v['labels'].values())
            purity = max_count / v['img_num']
            purities.append(purity)
            print(f'Person {name}: purity {purity:.4f} ({max_count}/{v["img_num"]})')
        print(f'Average purity: {np.mean(purities):.4f}')

        if prot_portion > 0:
            # 2. The quality of the clustering of protected images
            prot_num = len(protectee_prot_img_names)
            prot_preds = preds[orig_noise_num:]

            res = {}
            for idx in range(prot_num):
                label = prot_preds[idx]
                if label not in overall_res:
                    overall_res[label] = []
                overall_res[label].append(protectee_prot_img_names[idx])
                name = protectee_prot_img_names[idx].split('/')[-3]
                if name not in res:
                    res[name] = {}
                if label not in res[name]:
                    res[name][label] = 0
                res[name][label] += 1

            purities = []
            for name, v in res.items():
                max_count = max(v.values())
                purity = max_count / sum(v.values())
                purities.append(purity)
                print(f'Protected Person {name}: purity {purity:.4f} ({max_count}/{sum(v.values())})')
            print(f'Average protected purity: {np.mean(purities):.4f}')
        # Save clustering results
        save_path = os.path.join(BASE_PATH, 'results', 'cluster', f'{face_db_name}_{mask_name[0]}_{mask_name[1][:-4]}_{fr_name}_{f"prot{prot_portion*100:.0f}" if prot_portion > 0 else "clean"}.yaml')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(overall_res, f)