import os
from typing import List, Dict
import datetime

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yaml
from omegaconf import OmegaConf

from .FacialRecognition import FR
from .FaceDetection import FD
from .utils import load_imgs, load_mask, build_facedb, build_compressed_face_db, crop_face, complete_del, visualize_mask, eval_masks, compression_eval
from . import BASE_PATH
from .UVMapping import UVGenerator

def run(cfgs: OmegaConf, mode: str, data: Dict[str, Dict[str, List[str]]], train: callable = None) -> None:
    """
    Run training or evaluation based on the provided configuration.

    Args:
        cfgs (OmegaConf): Configuration object containing settings for the run.
        mode (str): Mode of operation, either 'train' or 'eval'.
        data (Dict[str, Dict[str, List[str]]]): Dictionary containing the paths to each image.
            {"Bradley_Cooper": {
                'train': [list of training image paths],
                'eval': [list of evaluation image paths]
            }, ...}
        train (callable, optional): Training function to be called if mode is 'train'.
    """
    assert mode in ['train', 'eval'], "Mode must be either 'train' or 'eval'."
    ####################################################################################################################
    # Set the paths
    ####################################################################################################################
    device = torch.device(cfgs.device)
    face_db_path = os.path.join(BASE_PATH, "face_db", cfgs.eval_db)
    noise_db_path = os.path.join(face_db_path, '_noise_db')
    exp_name = cfgs.exp_name
    res_base_path = os.path.join(BASE_PATH, 'results', 'eval') if mode == 'eval' else os.path.join(BASE_PATH, 'experiments')
    res_path = os.path.join(res_base_path, exp_name)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
        # print(f"The experiment results will be saved in {res_path}.")
    else:
        new_exp_name = f"{exp_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        cfgs.exp_name = new_exp_name
        # print(f"Experiment {exp_name} already exists. Changing it to {new_exp_name}")
        res_path = os.path.join(res_base_path, new_exp_name)
        os.makedirs(res_path)
    if mode == 'eval':
        mask_base_path = os.path.join(BASE_PATH, 'experiments', cfgs.mask_name[0])

    with torch.no_grad():
        ####################################################################################################################
        # Init SMIRK
        ####################################################################################################################
        smirk_base_path = os.path.join(BASE_PATH, 'smirk')
        smirk_weight_path = os.path.join(smirk_base_path, 'pretrained_models/SMIRK_em1.pt')
        mp_lmk_model_path = os.path.join(smirk_base_path, 'assets/face_landmarker.task')
        uvmapper = UVGenerator(smirk_ckpts_path=smirk_weight_path, smirk_base_path=smirk_base_path, mp_ldmk_model_path=mp_lmk_model_path, device=device)
        ####################################################################################################################
        # Init FD if needed
        ####################################################################################################################
        if cfgs.need_cropping:
            fd = FD(model_name=cfgs.fd_name, device=device)
    ####################################################################################################################
    # Init FRs
    ####################################################################################################################
    train_frs = [FR(model_name=fr_name, device=device) for fr_name in cfgs.train_fr_names]
    eval_frs = [FR(model_name=fr_name, device=device) for fr_name in cfgs.eval_fr_names]
    ####################################################################################################################
    # Main loop
    ####################################################################################################################
    training_times = []
    for protectee_idx, (protectee, protectee_data) in enumerate(data.items()):
        ####################################################################################################################
        # Enumerate through all the protectees
        ####################################################################################################################
        print("\n"+"#" * 50)
        print(f"Protectee: {protectee}")
        print("#" * 50+ "\n")
        res_save_path = os.path.join(res_path, protectee)
        os.makedirs(res_save_path, exist_ok=False)
        print(f"Created directory {res_save_path} for protectee {protectee}.")
        with open(os.path.join(res_save_path, 'cfgs.yaml'), 'w') as f:
                OmegaConf.save(cfgs, f)
        if mode == 'train':
            with torch.no_grad():
                if not cfgs.need_cropping:
                    training_imgs: torch.Tensor = load_imgs(img_paths=protectee_data['train'], img_sz=cfgs.mask_size, drange=255, device=device)
                    #print(training_imgs.shape, training_imgs.min(), training_imgs.max())
                else:
                    training_imgs: List[torch.Tensor] = load_imgs(img_paths=protectee_data['train'], img_sz=-1, drange=255, device=device)
                    _cropped_imgs = []
                    _no_face = []
                    for img_idx, img in enumerate(training_imgs):
                        cropped_face, pos = crop_face(img=img.squeeze(0), detector=fd, verbose=True)
                        if cropped_face is None or pos is None:
                            _no_face.append(img_idx)
                            continue
                        _cropped_imgs.append(cropped_face)
                    print(f"{len(_no_face)} training images do not have detectable faces and are ignored:")
                    for idx in _no_face:
                        print(f"{protectee_data['train'][idx]}")
                    training_imgs: torch.Tensor = torch.stack(_cropped_imgs, dim=0)
                    del _cropped_imgs
                    complete_del()
                #print(training_imgs.shape, training_imgs.min(), training_imgs.max())
                training_imgs.div_(255.)
                #print(training_imgs.shape, training_imgs.min(), training_imgs.max())
                img_num = training_imgs.shape[0]
                # Ensure UV mapping runs on the target device
                train_uvs, train_bin_masks, no_faces = uvmapper.forward(training_imgs,align_ldmks=cfgs.uv_gen_align_ldmk,batch_size=cfgs.uv_gen_batch)
                #print(training_imgs.shape, training_imgs.min(), training_imgs.max())
                if cfgs.uv_gen_align_ldmk:
                    training_imgs = [img for idx, img in enumerate(training_imgs) if idx not in no_faces]
                    train_uvs = [uv for idx, uv in enumerate(train_uvs) if idx not in no_faces]
                    train_bin_masks = [mask for idx, mask in enumerate(train_bin_masks) if idx not in no_faces]
                    training_imgs = torch.stack(training_imgs, dim=0)
                    train_uvs = torch.stack(train_uvs, dim=0)
                    train_bin_masks = torch.stack(train_bin_masks, dim=0)
                training_imgs = training_imgs.cpu()
                train_uvs = train_uvs.cpu()
                train_bin_masks = train_bin_masks.cpu()
                train_dl = DataLoader(
                    dataset=TensorDataset(training_imgs, train_uvs, train_bin_masks),
                    batch_size=cfgs.batch_size,
                    shuffle=cfgs.shuffle,
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True
                )
                del training_imgs, train_uvs, train_bin_masks
                complete_del()
            train_start = datetime.datetime.now()
            univ_mask: torch.Tensor = train(cfgs=cfgs, frs=train_frs, train_dl=train_dl, results_save_path=res_save_path)
            training_time = (datetime.datetime.now() - train_start).total_seconds()
            print(f"Training time for protectee {protectee}: {training_time:.2f} seconds.")
            training_times.append(training_time)
            if cfgs.save_univ_mask:
                np.save(os.path.join(res_save_path, "univ_mask.npy"), univ_mask.contiguous().numpy())
            if cfgs.visualize_interval > 0:
                for img_idx in range(0, img_num, cfgs.visualize_interval):
                    visualize_mask(
                        orig_img=train_dl.dataset[img_idx][0].to(device),
                        uv=train_dl.dataset[img_idx][1].to(device),
                        bin_mask=train_dl.dataset[img_idx][2].to(device),
                        univ_mask=univ_mask.clone().to(device),
                        epsilon=cfgs.epsilon,
                        save_path=os.path.join(res_save_path, f"train_vis_{img_idx}.png"),
                        use_bin_mask=cfgs.bin_mask,
                        three_d=cfgs.three_d)
            del train_dl
            complete_del()
        elif mode == 'eval':
            mask_path = os.path.join(mask_base_path, protectee, cfgs.mask_name[1])
            assert os.path.exists(mask_path), f"Mask file {mask_path} does not exist."
            univ_mask = load_mask(mask_path=mask_path, device=device)
        
        with torch.no_grad():
            eval_imgs = load_imgs(img_paths=protectee_data['eval'], img_sz=cfgs.mask_size, usage_portion=1.0, drange=255, device=device)
            if cfgs.need_cropping:
                _cropped_imgs = []
                _no_face = []
                for img in eval_imgs:
                    cropped_face, pos = crop_face(img=img, detector=fd, verbose=True)
                    if cropped_face is None or pos is None:
                        _no_face.append(img)
                        continue
                    _cropped_imgs.append(cropped_face)
                print(f"{len(_no_face)} eval images do not have detectable faces and are ignored:")
                for idx in _no_face:
                    print(f"{protectee_data['eval'][idx]}")
                eval_imgs = torch.stack(_cropped_imgs, dim=0)
            eval_imgs.div_(255.)
            img_num = eval_imgs.shape[0]
            # Ensure UV mapping runs on the target device, then move tensors back to CPU for DataLoader workers
            eval_uvs, eval_bin_masks, _ = uvmapper.forward(eval_imgs)
            eval_imgs = eval_imgs.cpu()
            eval_uvs = eval_uvs.cpu()
            eval_bin_masks = eval_bin_masks.cpu()
            workers = getattr(cfgs, 'num_workers', 4)
            eval_dl = DataLoader(
                dataset=TensorDataset(eval_imgs, eval_uvs, eval_bin_masks),
                batch_size=16,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )
            del eval_imgs, eval_uvs, eval_bin_masks
            complete_del()
            for eval_fr in eval_frs:
                noise_db = build_facedb(db_path=noise_db_path, fr_name=eval_fr.model_name, device=device)
                res = eval_masks(
                    three_d=cfgs.three_d,
                    test_data=eval_dl, 
                    face_db=noise_db,
                    fr=eval_fr,
                    device=device,
                    bin_mask=cfgs.bin_mask,
                    epsilon=cfgs.epsilon,
                    masks=univ_mask.repeat(img_num, 1, 1, 1), 
                    query_portion=cfgs.query_portion,
                    vis_eval=cfgs.vis_eval,
                    verbose=True
                )
                if mode == 'train':
                    res['training_time'] = training_time
                with open(os.path.join(res_save_path, f"eval_res_{eval_fr.model_name}.yaml"), 'w') as f:
                    yaml.dump(res, f)
                if cfgs.eval_compression:
                    noise_db = build_compressed_face_db(db_path=noise_db_path, 
                                                        fr=eval_fr, 
                                                        device=device, 
                                                        compression_methods=cfgs.eval_compression_methods, 
                                                        compression_cfgs=cfgs.compression_cfgs)
                    compression_res = compression_eval(
                        compression_methods=cfgs.eval_compression_methods, 
                        compression_cfgs=cfgs.compression_cfgs,
                        three_d=cfgs.three_d,
                        test_data=eval_dl,
                        face_db=noise_db,
                        fr=eval_fr,
                        device=device,
                        bin_mask=cfgs.bin_mask,
                        epsilon=cfgs.epsilon,
                        masks=univ_mask.repeat(img_num, 1, 1, 1),
                        query_portion=cfgs.query_portion
                    )
                    with open(os.path.join(res_save_path, f"compression_res_{eval_fr.model_name}.yaml"), 'w') as f:
                        yaml.dump(compression_res, f)
            del eval_dl
            complete_del()
    if mode == 'train':
        print(f"Average training time: {np.mean(training_times):.2f} seconds.")

                    