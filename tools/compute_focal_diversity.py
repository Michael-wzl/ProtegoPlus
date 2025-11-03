import os

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
    device = torch.device('cuda:6')
    model_pool = [n for n in BASIC_POOL if n not in ['ir50_adaface_casia']]#, 'ir50_arcface_casia', 'mobilefacenet_arcface_casia']]
    ensemble_sizes = [3]
    feature_base_paths = [f"{BASE_PATH}/face_db/face_scrub", f"{BASE_PATH}/face_db/face_scrub/_noise_db"]
    query_id_portion = 0.2
    query_img_portion = 0.5
    remove_common = False
    put_back = True
    standardize_size = -1  # Set to -1 for no standardization or any positive integer for fixed size
    allow_dup = False
    # Metric options:
    # - 'intersectional_size' (size of intersection of retrieved sets)
    # - 'jaccard_absrecall' (set-overlap + abs recall diff)
    # - 'top{k}classification' (majority label over top-k retrieved labels)
    # - 'soft_classification' (failure agreement weighted by per-query recall)
    # - 'soft_oracle' (soft OR-of-recalls across ensemble)
    # - 'soft_oracle_gain' (normalized gain over average model recall)
    # - 'failure_correlation' / 'failure_corr' (1 - avg Pearson corr of failures)
    # - 'dar' (diversity-adjusted recall = soft_oracle * (0.5 + 0.5*failure_corr))
    definition = 'performance_only'
    focal_diversity_save_base_path = f"{BASE_PATH}/results/focal_diversity"
    #save_intermediate_results = False
    intermediate_save_path = f"{BASE_PATH}/results/focal_diversity/retrieve_results.yaml"
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
            save_path = os.path.join(focal_diversity_save_base_path, f"focal_diversities_ens{ensemble_size}_{definition}.yaml")
            with open(save_path, 'w') as f:
                # Ensure values are native Python floats for YAML compatibility
                yaml.safe_dump({str(sorted(k)): float(v) for k, v in focal_diversities.items()}, f)
            print(f"Saved focal diversities to {save_path}")
    