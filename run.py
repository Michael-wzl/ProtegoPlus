import os
import datetime
import copy

import torch
import numpy as np
import yaml
from omegaconf import OmegaConf
import omegaconf

from FacialRecognition import *
from utils import *
from compression import compress

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

def run1(train: callable, cfgs: OmegaConf) -> None:
    ####################################################################################################################
    # Set the paths
    ####################################################################################################################
    device = torch.device(cfgs.device)
    face_db_path = os.path.join(BASE_PATH, "face_db", cfgs.eval_db)
    noise_db_path = os.path.join(face_db_path, '_noise_db')
    if isinstance(cfgs.protectees, str) and cfgs.protectees == 'all':
        protectees = [protectee for protectee in sorted(os.listdir(face_db_path)) if not protectee.startswith(('_', '.'))]
    elif isinstance(cfgs.protectees, int):
        protectees = [protectee for protectee in sorted(os.listdir(face_db_path)) if not protectee.startswith(('_', '.'))]
        if cfgs.protectees > 0:
            protectees = protectees[:cfgs.protectees]
            # print(f"Using the first {cfgs.protectees} protectees from the database.")
        elif cfgs.protectees < 0:
            protectees = protectees[cfgs.protectees:]
            # print(f"Using the last {-cfgs.protectees} protectees from the database.")
        else:
            raise ValueError("protectees cannot be 0.")
    elif isinstance(cfgs.protectees, omegaconf.listconfig.ListConfig):
        if cfgs.protectees[0] == '!':
            protectees = [protectee for protectee in sorted(os.listdir(face_db_path)) if not protectee.startswith(('_', '.')) and protectee not in cfgs.protectees[1:]]
        else:
            protectees = cfgs.protectees
    else:
        raise ValueError(f"Invalid type for protectees: {type(cfgs.protectees)}. Expected str, int, or list.")
    if ".DS_Store" in protectees:
        protectees.remove(".DS_Store")
    ####################################################################################################################
    # Create the folder for storing experiment results
    ####################################################################################################################
    exp_name = cfgs.exp_name
    ########################## ! Modified for public release ! ##########################
    res_base_path = os.path.join(BASE_PATH, 'results', 'eval')
    #!####################################################################################
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

    ####################################################################################################################
    # Main loop
    ####################################################################################################################
    training_time = []
    for protectee_idx, protectee in enumerate(protectees):
        print(f"Evaluating {protectee_idx+1}/{len(protectees)}: {protectee}")
        ####################################################################################################################
        # Enumerate through all the protectees
        ####################################################################################################################
        # print("\n"+"#" * 50)
        # print(f"Protectee: {protectee}")
        # print("#" * 50+ "\n")
        res_save_path = os.path.join(res_path, protectee)
        os.makedirs(res_save_path, exist_ok=False)
        # print(f"Created directory {res_save_path} for protectee {protectee}.")
        user_imgs = load_imgs(base_dir=os.path.join(face_db_path, protectee), img_sz=cfgs.mask_size, usage_portion=cfgs.usage_portion)
        user_uvs = load_uvs(base_dir=os.path.join(face_db_path, protectee), usage_portion=cfgs.usage_portion)
        user_bin_masks = load_bin_masks(base_dir=os.path.join(face_db_path, protectee), usage_portion=cfgs.usage_portion)
        
        train_dl, test_dl = split(imgs=user_imgs, uvs=user_uvs, bin_masks=user_bin_masks, train_portion=cfgs.train_portion, random_split=cfgs.random_split, batch_size=cfgs.batch_size, shuffle=cfgs.shuffle)
        # print(f"Split images for protectee {protectee}: train_dl: {len(train_dl.dataset)}; test_dl: {len(test_dl.dataset)}")
        for fr_pair_idx, (train_fr_names, test_fr_names) in enumerate(cfgs.fr_pairs):
            ####################################################################################################################
            # Enumerate through all the FR pairs
            ####################################################################################################################
            # print("$"*40)
            # print(f"FR Pair {fr_pair_idx+1}/{len(cfgs.fr_pairs)} with FR Pair {[train_fr_names, test_fr_names]} for protectee {protectee}.")
            # print("$"*40)
            train_frs = [FR(model_name=fr_name, device=device) for fr_name in train_fr_names]

            face_dbs = {test_fr_name: build_facedb(db_path=noise_db_path, fr_name=test_fr_name, device=device) for test_fr_name in test_fr_names}
            # print(f"Built face databases for test FRs: {list(face_dbs.keys())}")
            if cfgs.compression_eval:
                # print(f"Built compressed face databases for test FRs: {test_fr_names} with methods {cfgs.eval_compression_methods} and configs {cfgs.compression_cfgs}")
                compressed_face_dbs = {test_fr_name: build_compressed_face_db(db_path=noise_db_path, fr=FR(model_name=test_fr_name, device=device), device=device, compression_methods=cfgs.eval_compression_methods, compression_cfgs=cfgs.compression_cfgs) for test_fr_name in test_fr_names}
            ####################################################################################################################
            # Enumerate through all mask seeds. Used for testing the stability of the universal mask, relevant to the init seed.
            ####################################################################################################################
            for mask_idx, mask_seed in enumerate(cfgs.mask_seeds):
                # print("*"*40)
                # print(f"Mask seed {mask_idx+1}/{len(cfgs.mask_seeds)} with seed {mask_seed} for protectee {protectee}.")
                # print("*"*40)
                # Save the configurations for this run
                with open(os.path.join(res_save_path, f'frpair{fr_pair_idx}_mask{mask_idx}_config.yaml'), 'w') as f:
                    _cfgs = copy.deepcopy(cfgs)
                    _cfgs.fr_pairs = [train_fr_names, test_fr_names]
                    _cfgs.mask_seeds = mask_seed
                    # print("Configurations:")
                    # print(OmegaConf.to_container(_cfgs, resolve=True))
                    OmegaConf.save(_cfgs, f)
                # Start the training
                train_s_time = datetime.datetime.now()
                """train_res = train(
                    cfgs = cfgs, 
                    mask_random_seed=mask_seed,
                    frs = train_frs,
                    train_dl = train_dl,
                    test_dl = test_dl,
                    results_save_path=res_save_path
                )"""
                ########################## ! Modified for public release ! ##########################
                train_res = train(
                    cfgs = cfgs, 
                    protectee = protectee
                )
                #!####################################################################################
                univ_mask = train_res['univ']
                test_additional_masks = train_res['test']
                train_e_time = datetime.datetime.now()
                training_time.append((train_e_time - train_s_time).total_seconds())
                # print(f"Training time: {training_time[-1]} seconds")
                with torch.no_grad():
                    test_img_num = len(test_dl.dataset)
                    # print(univ_mask.shape, test_additional_masks.shape if test_additional_masks is not None else None)
                    univ_masks = np.repeat(univ_mask.copy(), test_img_num, axis=0) if univ_mask is not None else np.zeros((test_img_num, 3, cfgs.mask_size, cfgs.mask_size), dtype=np.float32)
                    test_additional_masks = test_additional_masks.copy() if test_additional_masks is not None else np.zeros((test_img_num, 3, cfgs.mask_size, cfgs.mask_size), dtype=np.float32)
                    test_masks = np.clip(univ_masks + test_additional_masks, -cfgs.epsilon, cfgs.epsilon)
                    ####################################################################################################################
                    # Save and visualize the test_masks
                    ####################################################################################################################
                    if cfgs.save_univ_mask: 
                        np.save(os.path.join(res_save_path, f'frpair{fr_pair_idx}_mask{mask_idx}_univ_mask.npy'), univ_masks)
                    if cfgs.save_additional_mask_interval > 0 or cfgs.visualize_interval > 0: 
                        for img_idx, additional_mask in enumerate(test_additional_masks):
                            if cfgs.save_additional_mask_interval > 0 and img_idx % cfgs.save_additional_mask_interval == 0:
                                np.save(os.path.join(res_save_path, f'frpair{fr_pair_idx}_additional_mask{mask_idx}_img{img_idx}.npy'), additional_mask)
                            if cfgs.visualize_interval > 0 and img_idx % cfgs.visualize_interval == 0:
                                _orig_img = test_dl.dataset[img_idx][0].clone().unsqueeze(0).cpu()
                                _uv = test_dl.dataset[img_idx][1].clone().unsqueeze(0).cpu()
                                _bin_mask = test_dl.dataset[img_idx][2].clone().unsqueeze(0).cpu() if cfgs.bin_mask else None
                                _additional_mask = torch.tensor(np.expand_dims(additional_mask.copy(), axis=0))
                                _univ_mask = torch.tensor(univ_masks[img_idx:img_idx+1, :, :, :].copy(), dtype=torch.float32)
                                save_name = f'frpair{fr_pair_idx}_mask{mask_idx}_img{img_idx}_visualize_mask.png'
                                if cfgs.three_d:
                                    visualize_3dmask(orig_img=_orig_img,
                                                    uv=_uv,
                                                    bin_mask=_bin_mask, 
                                                    save_path=os.path.join(res_save_path, save_name),
                                                    univ_texture=_univ_mask,
                                                    additional_texture=_additional_mask, 
                                                    epsilon=cfgs.epsilon
                                                    )
                                else:
                                    visualize_2dmask(orig_img=_orig_img,
                                                    univ_mask=_univ_mask,
                                                    bin_mask=_bin_mask,
                                                    additional_mask=_additional_mask,
                                                    save_path=os.path.join(res_save_path, save_name),
                                                    epsilon=cfgs.epsilon
                                                    )
                    for test_fr_name in test_fr_names:
                        ####################################################################################################################
                        # Enumerate through all the test FRs
                        ####################################################################################################################
                        # print(f"\n Evaluating with {test_fr_name}...")
                        test_fr = FR(model_name=test_fr_name, device=device)
                        face_db = copy.deepcopy(face_dbs[test_fr_name])
                        res = eval_masks(
                            three_d=cfgs.three_d,
                            test_data=copy.deepcopy(test_dl),
                            face_db=face_db,
                            fr=test_fr,
                            device=device,
                            bin_mask=cfgs.bin_mask,
                            epsilon=cfgs.epsilon,
                            masks=test_masks.copy(), 
                            query_portion=cfgs.query_portion, 
                            vis_eval=cfgs.vis_eval
                        )
                        with open(os.path.join(res_save_path, f'frpair{fr_pair_idx}_mask{mask_idx}_testfr{test_fr_name}_results.yaml'), 'w') as f:
                            yaml.dump(res, f)
                        if cfgs.compression_eval:
                            compression_res = compression_eval(
                                compression_methods=cfgs.eval_compression_methods, 
                                compression_cfgs=cfgs.compression_cfgs,
                                three_d=cfgs.three_d,
                                test_data=copy.deepcopy(test_dl),
                                face_db=compressed_face_dbs[test_fr_name],
                                fr=test_fr,
                                device=device,
                                bin_mask=cfgs.bin_mask,
                                epsilon=cfgs.epsilon,
                                masks=test_masks.copy(), 
                                query_portion=cfgs.query_portion
                            )
                            with open(os.path.join(res_save_path, f'frpair{fr_pair_idx}_mask{mask_idx}_testfr{test_fr_name}_compression_results.yaml'), 'w') as f:
                                yaml.dump(compression_res, f)
    # print(f"Mean Training time: {np.mean(training_time)} seconds, Max Training time: {np.max(training_time)} seconds, Min Training time: {np.min(training_time)} seconds")
    
def run2(train: callable, cfgs: OmegaConf) -> None:
    ####################################################################################################################
    # Set the paths
    ####################################################################################################################
    device = torch.device(cfgs.device)
    face_db_path = os.path.join(BASE_PATH, "face_db", cfgs.eval_db)
    noise_db_path = os.path.join(face_db_path, '_noise_db')
    if isinstance(cfgs.protectees, str) and cfgs.protectees == 'all':
        protectees = [protectee for protectee in sorted(os.listdir(face_db_path)) if not protectee.startswith(('_', '.'))]
    elif isinstance(cfgs.protectees, int):
        protectees = [protectee for protectee in sorted(os.listdir(face_db_path)) if not protectee.startswith(('_', '.'))]
        if cfgs.protectees > 0:
            protectees = protectees[:cfgs.protectees]
            # print(f"Using the first {cfgs.protectees} protectees from the database.")
        elif cfgs.protectees < 0:
            protectees = protectees[cfgs.protectees:]
            # print(f"Using the last {-cfgs.protectees} protectees from the database.")
        else:
            raise ValueError("protectees cannot be 0.")
    elif isinstance(cfgs.protectees, omegaconf.listconfig.ListConfig):
        if cfgs.protectees[0] == '!':
            protectees = [protectee for protectee in sorted(os.listdir(face_db_path)) if not protectee.startswith(('_', '.')) and protectee not in cfgs.protectees[1:]]
        else:
            protectees = cfgs.protectees
    else:
        raise ValueError(f"Invalid type for protectees: {type(cfgs.protectees)}. Expected str, int, or list.")
    if ".DS_Store" in protectees:
        protectees.remove(".DS_Store")

    ####################################################################################################################
    # Create the folder for storing experiment results
    ####################################################################################################################
    exp_name = cfgs.exp_name
    ########################## ! Modified for public release ! ##########################
    res_base_path = os.path.join(BASE_PATH, 'results', 'eval')
    #!####################################################################################
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

    ####################################################################################################################
    # Main loop
    ####################################################################################################################

    #!###################################################################################################################
    user_data = {}
    #!###################################################################################################################
    training_time = []
    for protectee_idx, protectee in enumerate(protectees):
        """if protectee not in ['Matt_Damon', 'Matthew_Perry', 'Michael_Weatherly', 'Milo_Ventimiglia', 'Miranda_Cosgrove', 'Sarah_Hyland', 'Sean_Bean', 'Tina_Fey']:
            continue"""
        #!###################################################################################################################
        # ! Init the recorder
        #!###################################################################################################################
        user_data[protectee] = {
            "orig_features": {
                'orig': {},  
                **{compression: {} for compression in cfgs.eval_compression_methods} 
            },
            'protected_features': {}
        }
        
        for fr_pair_idx in range(len(cfgs.fr_pairs)):
            user_data[protectee]['protected_features'][fr_pair_idx] = {}
            for mask_idx in range(len(cfgs.mask_seeds)):
                user_data[protectee]['protected_features'][fr_pair_idx][mask_idx] = {
                    'orig': {},
                    **{compression: {} for compression in cfgs.eval_compression_methods} 
                }

        ####################################################################################################################
        # Enumerate through all the protectees
        ####################################################################################################################
        # print("\n"+"#" * 50)
        # print(f"Protectee: {protectee}")
        # print("#" * 50+ "\n")
        res_save_path = os.path.join(res_path, protectee)
        os.makedirs(res_save_path, exist_ok=False)
        # print(f"Created directory {res_save_path} for protectee {protectee}.")
        user_imgs = load_imgs(base_dir=os.path.join(face_db_path, protectee), img_sz=cfgs.mask_size, usage_portion=cfgs.usage_portion)
        user_uvs = load_uvs(base_dir=os.path.join(face_db_path, protectee), usage_portion=cfgs.usage_portion)
        user_bin_masks = load_bin_masks(base_dir=os.path.join(face_db_path, protectee), usage_portion=cfgs.usage_portion)

        train_dl, test_dl = split(imgs=user_imgs, uvs=user_uvs, bin_masks=user_bin_masks, train_portion=cfgs.train_portion, random_split=cfgs.random_split, batch_size=cfgs.batch_size, shuffle=cfgs.shuffle)
        #!###################################################################################################################
        #! Save the original and compressed features for the protectee for future evaluation
        #!###################################################################################################################
        with torch.no_grad():
            for fr_pair_idx, (train_fr_names, test_fr_names) in enumerate(cfgs.fr_pairs):
                for test_fr_name in test_fr_names:
                    test_fr = FR(model_name=test_fr_name, device=device)
                    orig_features = []
                    compressed_features = {}
                    for tensors in test_dl:
                        orig_features.append(test_fr(tensors[0].to(device)).cpu())
                        if cfgs.compression_eval:
                            for compression in cfgs.eval_compression_methods:
                                if compression not in compressed_features:
                                    compressed_features[compression] = []
                                compressed_features[compression].append(test_fr(compress(tensors[0].to(device), method=compression, differentiable=False, **cfgs.compression_cfgs[compression])).cpu())
                    orig_features = torch.cat(orig_features, dim=0)
                    user_data[protectee]['orig_features']['orig'][test_fr_name] = orig_features
                    if cfgs.compression_eval:
                        for compression in cfgs.eval_compression_methods:
                            compressed_features[compression] = torch.cat(compressed_features[compression], dim=0)
                            user_data[protectee]['orig_features'][compression][test_fr_name] = copy.deepcopy(compressed_features[compression])
        #!###################################################################################################################

        # print(f"Split images for protectee {protectee}: train_dl: {len(train_dl.dataset)}; test_dl: {len(test_dl.dataset)}")
        for fr_pair_idx, (train_fr_names, test_fr_names) in enumerate(cfgs.fr_pairs):
            ####################################################################################################################
            # Enumerate through all the FR pairs
            ####################################################################################################################
            # print("$"*40)
            # print(f"FR Pair {fr_pair_idx+1}/{len(cfgs.fr_pairs)} with FR Pair {[train_fr_names, test_fr_names]} for protectee {protectee}.")
            # print("$"*40)
            train_frs = [FR(model_name=fr_name, device=device) for fr_name in train_fr_names]

            face_dbs = {test_fr_name: build_facedb(db_path=noise_db_path, fr_name=test_fr_name, device=device) for test_fr_name in test_fr_names}
            # print(f"Built face databases for test FRs: {list(face_dbs.keys())}")
            ####################################################################################################################
            # Enumerate through all mask seeds. Used for testing the stability of the universal mask, relevant to the init seed.
            ####################################################################################################################
            for mask_idx, mask_seed in enumerate(cfgs.mask_seeds):
                # print("*"*40)
                # print(f"Mask seed {mask_idx+1}/{len(cfgs.mask_seeds)} with seed {mask_seed} for protectee {protectee}.")
                # print("*"*40)
                # Save the configurations for this run
                with open(os.path.join(res_save_path, f'frpair{fr_pair_idx}_mask{mask_idx}_config.yaml'), 'w') as f:
                    _cfgs = copy.deepcopy(cfgs)
                    _cfgs.fr_pairs = [train_fr_names, test_fr_names]
                    _cfgs.mask_seeds = mask_seed
                    # print("Configurations:")
                    # print(OmegaConf.to_container(_cfgs, resolve=True))
                    OmegaConf.save(_cfgs, f)
                # Start the training
                train_s_time = datetime.datetime.now()
                """train_res = train(
                    cfgs = cfgs, 
                    mask_random_seed=mask_seed,
                    frs = train_frs,
                    train_dl = train_dl,
                    test_dl = test_dl,
                    results_save_path=res_save_path
                )"""
                ########################## ! Modified for public release ! ##########################
                train_res = train(
                    cfgs = cfgs, 
                    protectee = protectee
                )
                #!####################################################################################
                univ_mask = train_res['univ']
                test_additional_masks = train_res['test']
                train_e_time = datetime.datetime.now()
                training_time.append((train_e_time - train_s_time).total_seconds())
                # print(f"Training time: {training_time[-1]} seconds")
                with torch.no_grad():
                    test_img_num = len(test_dl.dataset)
                    # print(univ_mask.shape, test_additional_masks.shape if test_additional_masks is not None else None)
                    univ_masks = np.repeat(univ_mask.copy(), test_img_num, axis=0) if univ_mask is not None else np.zeros((test_img_num, 3, cfgs.mask_size, cfgs.mask_size), dtype=np.float32)
                    test_additional_masks = test_additional_masks.copy() if test_additional_masks is not None else np.zeros((test_img_num, 3, cfgs.mask_size, cfgs.mask_size), dtype=np.float32)
                    test_masks = np.clip(univ_masks + test_additional_masks, -cfgs.epsilon, cfgs.epsilon)
                    ####################################################################################################################
                    # Save and visualize the test_masks
                    ####################################################################################################################
                    if cfgs.save_univ_mask: 
                        np.save(os.path.join(res_save_path, f'frpair{fr_pair_idx}_mask{mask_idx}_univ_mask.npy'), univ_masks)
                    if cfgs.save_additional_mask_interval > 0 or cfgs.visualize_interval > 0: 
                        for img_idx, additional_mask in enumerate(test_additional_masks):
                            if cfgs.save_additional_mask_interval > 0 and img_idx % cfgs.save_additional_mask_interval == 0:
                                np.save(os.path.join(res_save_path, f'frpair{fr_pair_idx}_additional_mask{mask_idx}_img{img_idx}.npy'), additional_mask)
                            if cfgs.visualize_interval > 0 and img_idx % cfgs.visualize_interval == 0:
                                _orig_img = test_dl.dataset[img_idx][0].clone().unsqueeze(0).cpu()
                                _uv = test_dl.dataset[img_idx][1].clone().unsqueeze(0).cpu()
                                _bin_mask = test_dl.dataset[img_idx][2].clone().unsqueeze(0).cpu() if cfgs.bin_mask else None
                                _additional_mask = torch.tensor(np.expand_dims(additional_mask.copy(), axis=0))
                                _univ_mask = torch.tensor(univ_masks[img_idx:img_idx+1, :, :, :].copy(), dtype=torch.float32)
                                save_name = f'frpair{fr_pair_idx}_mask{mask_idx}_img{img_idx}_visualize_mask.png'
                                if cfgs.three_d:
                                    visualize_3dmask(orig_img=_orig_img,
                                                    uv=_uv,
                                                    bin_mask=_bin_mask, 
                                                    save_path=os.path.join(res_save_path, save_name),
                                                    univ_texture=_univ_mask,
                                                    additional_texture=_additional_mask, 
                                                    epsilon=cfgs.epsilon
                                                    )
                                else:
                                    visualize_2dmask(orig_img=_orig_img,
                                                    univ_mask=_univ_mask,
                                                    bin_mask=_bin_mask,
                                                    additional_mask=_additional_mask,
                                                    save_path=os.path.join(res_save_path, save_name),
                                                    epsilon=cfgs.epsilon
                                                    )
                    for test_fr_name in test_fr_names:
                        ####################################################################################################################
                        # Enumerate through all the test FRs
                        ####################################################################################################################
                        # print(f"\n Evaluating with {test_fr_name}...")
                        test_fr = FR(model_name=test_fr_name, device=device)
                        face_db = copy.deepcopy(face_dbs[test_fr_name])
                        res = eval_masks(
                            three_d=cfgs.three_d,
                            test_data=copy.deepcopy(test_dl),
                            face_db=face_db,
                            fr=test_fr,
                            device=device,
                            bin_mask=cfgs.bin_mask,
                            epsilon=cfgs.epsilon,
                            masks=test_masks.copy(), 
                            query_portion=cfgs.query_portion, 
                            vis_eval=cfgs.vis_eval
                        )
                        with open(os.path.join(res_save_path, f'frpair{fr_pair_idx}_mask{mask_idx}_testfr{test_fr_name}_results.yaml'), 'w') as f:
                            yaml.dump(res, f)
                    #!###################################################################################################################
                    #! Save the protected features for the protectee for future evaluation
                    #!###################################################################################################################
                    protected_test_imgs = get_protected_imgs(
                        masks=torch.tensor(test_masks.copy(), dtype=torch.float32, device=device), 
                        dl=test_dl, 
                        epsilon=cfgs.epsilon, 
                        device=device, 
                        three_d=cfgs.three_d, 
                        bin_mask=cfgs.bin_mask
                    )
                    for test_fr_name in test_fr_names:
                        test_fr = FR(model_name=test_fr_name, device=device)
                        protected_test_features = test_fr(protected_test_imgs)
                        user_data[protectee]['protected_features'][fr_pair_idx][mask_idx]['orig'][test_fr_name] = protected_test_features.cpu()
                        if cfgs.compression_eval:
                            for compression in cfgs.eval_compression_methods:
                                protected_test_features_compressed = test_fr(compress(protected_test_imgs, method=compression, differentiable=False, **cfgs.compression_cfgs[compression]))
                                user_data[protectee]['protected_features'][fr_pair_idx][mask_idx][compression][test_fr_name] = protected_test_features_compressed.cpu()
                    #!#####################################################################################################################
    #!###################################################################################################################
    #! Evaluate the protected features for the protectee
    #!###################################################################################################################
    for protectee_idx, protectee in enumerate(protectees):
        print(f"Evaluating {protectee_idx+1}/{len(protectees)}: {protectee}")
        # print("\n"+"#" * 50)
        # print(f"Evaluating Protectee: {protectee}")
        # print("#" * 50+ "\n")
        res_save_path = os.path.join(res_path, protectee)
        for fr_pair_idx, (train_fr_names, test_fr_names) in enumerate(cfgs.fr_pairs):
            face_dbs = {test_fr_name: build_facedb(db_path=noise_db_path, fr_name=test_fr_name, device=device) for test_fr_name in test_fr_names}
            if cfgs.compression_eval:
                compressed_face_dbs = {test_fr_name: build_compressed_face_db(db_path=noise_db_path, fr=FR(model_name=test_fr_name, device=device), device=device, compression_methods=cfgs.eval_compression_methods, compression_cfgs=cfgs.compression_cfgs) for test_fr_name in test_fr_names}
            for mask_idx, mask_seed in enumerate(cfgs.mask_seeds):
                # print(f"Evaluating FR Pair {fr_pair_idx+1}/{len(cfgs.fr_pairs)} with Mask Seed {mask_idx+1}/{len(cfgs.mask_seeds)} for Protectee {protectee}.")
                for test_fr_name in test_fr_names:
                    # 1A and 2A
                    face_db = copy.deepcopy(face_dbs[test_fr_name])
                    face_db = {K: v.to(device) for K, v in face_db.items()}  # Ensure the face_db is on the correct device
                    for user_name, v in user_data.items():
                        if user_name == protectee:
                            continue
                        _protected_features = v['protected_features'][fr_pair_idx][mask_idx]['orig'][test_fr_name]
                        face_db[user_name] = _protected_features.to(device)
                    test_orig_features = user_data[protectee]['orig_features']['orig'][test_fr_name].clone()
                    test_protected_features = user_data[protectee]['protected_features'][fr_pair_idx][mask_idx]['orig'][test_fr_name].clone()
                    res = prot_eval(
                        orig_features=test_orig_features,
                        protected_features=test_protected_features,
                        face_db=face_db,
                        dist_func='cosine',
                        query_portion=cfgs.query_portion,
                        device=device
                    )
                    ona_a = res['1a']
                    two_a = res['2a']

                    # 1B and 2B
                    face_db = copy.deepcopy(face_dbs[test_fr_name])
                    face_db = {K: v.to(device) for K, v in face_db.items()}  # Ensure the face_db is on the correct device
                    for user_name, v in user_data.items():
                        if user_name == protectee:
                            continue
                        _orig_features = v['orig_features']['orig'][test_fr_name]
                        face_db[user_name] = _orig_features.to(device)
                    test_orig_features = user_data[protectee]['orig_features']['orig'][test_fr_name].clone()
                    test_protected_features = user_data[protectee]['protected_features'][fr_pair_idx][mask_idx]['orig'][test_fr_name].clone()
                    res = prot_eval(
                        orig_features=test_orig_features,
                        protected_features=test_protected_features,
                        face_db=face_db,
                        dist_func='cosine',
                        query_portion=cfgs.query_portion,
                        device=device
                    )
                    one_b = res['1b']
                    two_b = res['2b']

                    with open(os.path.join(res_save_path, f'frpair{fr_pair_idx}_mask{mask_idx}_testfr{test_fr_name}_results.yaml'), 'r') as f:
                        res = yaml.safe_load(f)
                        res['1a'] = ona_a
                        res['1b'] = one_b
                        res['2a'] = two_a
                        res['2b'] = two_b
                    with open(os.path.join(res_save_path, f'frpair{fr_pair_idx}_mask{mask_idx}_testfr{test_fr_name}_results.yaml'), 'w') as f:
                        yaml.dump(res, f)

                    if cfgs.compression_eval:
                        compression_res = {}
                        for compression in cfgs.eval_compression_methods:
                            # 1A and 2A
                            face_db = copy.deepcopy(compressed_face_dbs[test_fr_name][compression])
                            face_db = {K: v.to(device) for K, v in face_db.items()}  # Ensure the face_db is on the correct device
                            for user_name, v in user_data.items():
                                if user_name == protectee:
                                    continue
                                _protected_features = v['protected_features'][fr_pair_idx][mask_idx][compression][test_fr_name]
                                face_db[user_name] = _protected_features.to(device)
                            test_orig_features = user_data[protectee]['orig_features'][compression][test_fr_name].clone()
                            test_protected_features = user_data[protectee]['protected_features'][fr_pair_idx][mask_idx][compression][test_fr_name].clone()
                            res = prot_eval(
                                orig_features=test_orig_features,
                                protected_features=test_protected_features,
                                face_db=face_db,
                                dist_func='cosine',
                                query_portion=cfgs.query_portion,
                                device=device
                            )
                            ona_a = res['1a']
                            two_a = res['2a']

                            # 1B and 2B
                            face_db = copy.deepcopy(compressed_face_dbs[test_fr_name][compression])
                            face_db = {K: v.to(device) for K, v in face_db.items()}  # Ensure the face_db is on the correct device
                            for user_name, v in user_data.items():
                                if user_name == protectee:
                                    continue
                                _orig_features = v['orig_features'][compression][test_fr_name]
                                face_db[user_name] = _orig_features.to(device)
                            test_orig_features = user_data[protectee]['orig_features'][compression][test_fr_name].clone()
                            test_protected_features = user_data[protectee]['protected_features'][fr_pair_idx][mask_idx][compression][test_fr_name].clone()
                            res = prot_eval(
                                orig_features=test_orig_features,
                                protected_features=test_protected_features,
                                face_db=face_db,
                                dist_func='cosine',
                                query_portion=cfgs.query_portion,
                                device=device
                            )
                            one_b = res['1b']
                            two_b = res['2b']

                            compression_res[compression] = {
                                '1a': ona_a,
                                '1b': one_b,
                                '2a': two_a,
                                '2b': two_b
                            }
                        with open(os.path.join(res_save_path, f'frpair{fr_pair_idx}_mask{mask_idx}_testfr{test_fr_name}_compression_results.yaml'), 'w') as f:
                            yaml.dump(compression_res, f)
    #!###################################################################################################################
    # print(f"Mean Training time: {np.mean(training_time)} seconds, Max Training time: {np.max(training_time)} seconds, Min Training time: {np.min(training_time)} seconds")

def get_protected_imgs(masks: torch.Tensor, 
                       dl: DataLoader, 
                       epsilon: float, 
                       device: torch.device, 
                       three_d: bool = True, 
                       bin_mask: bool = False) -> torch.Tensor:
    img_cnt = 0
    protected_imgs = []
    for tensors in dl:
        imgs = tensors[0].to(device)
        uvs = tensors[1].to(device)
        if bin_mask:
            bin_masks = tensors[2].to(device)
        if three_d:
            perts = torch.clamp(F.grid_sample(masks[img_cnt:img_cnt+len(tensors[0]), :, :, :], uvs, align_corners=True, mode='bilinear'), -epsilon, epsilon)
        else:
            perts = masks[img_cnt:img_cnt+len(tensors[0]), :, :, :]
        if bin_mask:
            perts *= bin_masks
        img_cnt += len(tensors[0])
        protected_imgs.append(torch.clamp(imgs + perts, 0, 1))
    protected_imgs = torch.cat(protected_imgs, dim=0)
    return protected_imgs
