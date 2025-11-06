import os
import argparse
import datetime

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
import yaml

from protego import BASE_PATH
from protego.FacialRecognition import BASIC_POOL, VIT_FAMILY, SPECIAL_POOL
from protego.focal_diversity import get_focal_diversities

if __name__ == "__main__":
    ######################### Configuration #########################
    parser = argparse.ArgumentParser(description="Compute focal diversity")
    parser.add_argument('--device', type=str, default="cuda:0", help='device to use')
    parser.add_argument('--exp_name', type=str, default="default_exp", help='experiment name for saving results')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    model_pool = [n for n in BASIC_POOL if n not in ['ir50_adaface_casia']]#, 'ir50_arcface_casia', 'mobilefacenet_arcface_casia']]
    ensemble_sizes = [3] # Only 3 and 4 are supported for now
    feature_base_paths = [f"{BASE_PATH}/face_db/face_scrub", f"{BASE_PATH}/face_db/face_scrub/_noise_db"]
    query_id_portion = 0.2
    query_img_portion = 0.5
    remove_common = False
    put_back = True
    standardize_size = -1  # Set to -1 for no standardization or any positive integer for fixed size
    allow_dup = False
    definition = 'performance_only'
    focal_diversity_save_base_path = f"{BASE_PATH}/results/focal_diversity/{args.exp_name}"
    if os.path.exists(focal_diversity_save_base_path):
        print(f"Focal diversity save path {focal_diversity_save_base_path} already exists.")
        new_name = f"{args.exp_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        focal_diversity_save_base_path = f"{BASE_PATH}/results/focal_diversity/{new_name}"
        print(f"Changing experiment name to {new_name} and save path to {focal_diversity_save_base_path}")
    os.makedirs(focal_diversity_save_base_path, exist_ok=False)
    #save_intermediate_results = False
    #intermediate_save_path = f"{BASE_PATH}/results/focal_diversity/retrieve_results.yaml"
    ###############################################################
    with torch.no_grad():
        for size_idx, ensemble_size in enumerate(ensemble_sizes):
            print(f"Computing focal diversities for ensemble size {ensemble_size} ({size_idx+1}/{len(ensemble_sizes)})")
            results = get_focal_diversities(
                model_pool=model_pool, 
                ensemble_size=ensemble_size, 
                feature_base_paths=feature_base_paths, 
                query_id_portion=query_id_portion,
                query_img_portion=query_img_portion,
                standardize_size=standardize_size,
                allow_dup=allow_dup,
                definition=definition, 
                remove_common=remove_common,
                put_back=put_back,
                device=device, 
                verbose=True
            )
            focal_diversities = results['focal_diversities']
            retrieve_results = results['retrieval_results']
            for fr_names, focal_div in focal_diversities.items():
                print(f"FR Models: {fr_names}, Focal Diversity ({definition}): {focal_div:.4f}")
            """if save_intermediate_results:
                with open(intermediate_save_path, 'w') as f:
                    yaml.safe_dump(retrieve_results, f)
                print(f"Saved retrieval results to {intermediate_save_path}")"""
            save_fname = f"focal_diversities_ens{ensemble_size}_{definition}.yaml"
            save_path = os.path.join(focal_diversity_save_base_path, save_fname)
            with open(save_path, 'w') as f:
                # Ensure values are native Python floats for YAML compatibility
                yaml.safe_dump({str(sorted(k)): float(v) for k, v in focal_diversities.items()}, f)
            print(f"Saved focal diversities to {save_path}")
            # Run analyze_focal_diversity.py automatically after computing focal diversities
            analyze_script = f"tools.analyze_focal_diversity"
            cmd = f"python3 -m {analyze_script} --exp_name {args.exp_name} --fd_fname {save_fname} --ensemble_size {ensemble_size}"
            print(f"Running analysis script: {cmd}")
            os.system(cmd)
    