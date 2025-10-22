import os
from itertools import combinations
from typing import Dict, List, Tuple

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
import yaml

from protego.FacialRecognition import FR, BASIC_POOL, VIT_FAMILY, SPECIAL_POOL
from protego.utils import retrieve, load_imgs

def get_negative_samples(face_db: Dict[str, Tuple[List[str], torch.Tensor]], fr: FR, device: torch.device, query_portion: float = 0.5) -> List[str]:
    """
    Identify negative samples (failures) for a given FR model using a strict Top-1 failure rule.

    Definition used here (closer to standard FR evaluation and the paper's wording):
    - A query is counted as a negative sample if its Top-1 retrieval does NOT match the true identity.

    Implementation details:
    - For each identity (name), we split its images into queries (first portion) and gallery (remaining portion).
    - The database is the union of all other identities' features plus the gallery portion of the same identity.
    - We run retrieval with topk=1; retrieve() returns the fraction of correct matches among top-k.
      With k=1, accu ∈ {0.0, 1.0}. We mark it negative iff accu == 0.0.

    Returns a list of image paths (identifiers) that are negatives for this model.
    """
    features_dict = {}
    imgs_paths = []
    for name, (imgs_path, imgs) in face_db.items():
        features_dict[name] = fr(imgs.to(device)).cpu()
        imgs = imgs.cpu()
        imgs_paths.extend(imgs_path)
    negative_samples = []
    start_idx = 0
    for name, features in features_dict.items():
        db, db_labels = [], []
        for n, f in features_dict.items():
            if n != name:
                db.append(f)
                db_labels.extend([n] * f.shape[0])
        db = torch.cat(db, dim=0).to(device)
        feature_nums = features.shape[0]
        query_nums = int(feature_nums * query_portion)
        queries = features[:query_nums].to(device)
        query_labels = [name] * query_nums
        db = torch.cat([db, features[query_nums:].to(device)], dim=0)
        db_labels.extend([name] * (feature_nums - query_nums))
        # Strict Top-1 failure: topk=1 and negative if accu == 0.0
        res = retrieve(
            db=db,
            db_labels=db_labels,
            queries=queries,
            query_labels=query_labels,
            dist_func='cosine',
            topk=1,
        )
        for idx, accu in enumerate(res):
            if accu == 0.0:  # Top-1 miss
                negative_samples.append(imgs_paths[start_idx + idx])
        start_idx += feature_nums
    return negative_samples

def cal_focal_diversity(ensemble: List[str], model_results: Dict[str, List[str]]) -> float:
    """
    Compute focal diversity for a given ensemble using the paper's definition.

    Inputs
    - ensemble: list of model names (size S)
    - model_results: mapping model_name -> list of negative sample identifiers (e.g., image paths)

    Method (faithful-but-practical implementation):
    - For each focal model F in the ensemble, gather its negative set N_F.
    - For every sample x in N_F, check how many of the remaining (S-1) models also fail on x.
      Let agreement_on_x = (# of other models that also fail on x) / max(1, S-1).
    - Define λ_focal(T; F) as the average agreement_on_x over x in N_F.
      Intuition: λ measures agreement (correlation) of failures; 1-λ is disagreement/diversity.
        - The focal diversity is d_focal(T) = average_{F in T with |N_F|>0} [1 - λ_focal(T; F)].
            That is, we average only over focal members that have at least one negative sample
            (S_eff), which avoids biasing the score downward when a very strong model has no
            negatives and provides no evidence about failure correlation.

    Edge cases
        - If a focal model has no negative samples, we skip it from the averaging set
            (no evidence to estimate correlation). If none have negatives, return 0.
    - If ensemble size S <= 1, return 0.0.

    Returns
    - Scalar focal diversity in [0, 1]. Higher means more diverse (more decorrelated failures).
    """
    S = len(ensemble)
    if S <= 1:
        return 0.0

    # Build sets for O(1) lookups; ignore models not present in model_results gracefully
    neg_sets: Dict[str, set] = {}
    for m in ensemble:
        neg_sets[m] = set(model_results.get(m, []))

    total_div = 0.0
    eff_count = 0  # number of focal members with at least one negative
    for focal in ensemble:
        negatives = neg_sets.get(focal, set())
        if not negatives:
            # No negatives for focal model -> skip from averaging
            continue

        agree_sum = 0.0
        denom_models = max(1, S - 1)
        for sample in negatives:
            # Count how many other models also fail on this sample
            also_fail = 0
            for other in ensemble:
                if other == focal:
                    continue
                if sample in neg_sets[other]:
                    also_fail += 1
            agreement_frac = also_fail / denom_models
            agree_sum += agreement_frac

        # λ_focal: average agreement across focal's negatives
        lambda_focal = agree_sum / max(1, len(negatives))
        total_div += (1.0 - lambda_focal)
        eff_count += 1

    # Average across focal members that have negatives (S_eff). If none, return 0.
    if eff_count == 0:
        return 0.0
    d_focal = total_div / eff_count
    # Clamp to [0,1] for numerical hygiene
    d_focal = max(0.0, min(1.0, d_focal))
    return d_focal

if __name__ == "__main__":
    ####################################################################################################################
    # Configuration
	####################################################################################################################
    with torch.no_grad():
        face_db_base_path = "/home/zlwang/ProtegoPlus/face_db/face_scrub"
        device = torch.device("cuda:0")
        all_fr_names = [n for n in BASIC_POOL + SPECIAL_POOL if n not in VIT_FAMILY]
        ensemble_sizes = [i for i in range(2, len(all_fr_names) + 1)]
        negative_sample_res_path = "/home/zlwang/ProtegoPlus/results/eval/negative_samples.yaml"
        overwrite = True
        diversity_res_path = "/home/zlwang/ProtegoPlus/results/eval/focal_diversity.yaml"
        ####################################################################################################################
        if overwrite:
            db_names = [os.path.join(face_db_base_path, n) for n in os.listdir(face_db_base_path) if not n.startswith(('.', '_'))]
            db_names.extend([os.path.join(face_db_base_path, '_noise_db', n) for n in os.listdir(os.path.join(face_db_base_path, '_noise_db')) if not n.startswith(('.', '_'))])
            face_db, img_paths = {}, []
            for name in db_names:
                imgs, personal_img_paths = load_imgs(base_dir=name, img_sz=224, usage_portion=1., drange=1, device=torch.device("cpu"), return_img_paths=True)
                face_db[name] = (personal_img_paths, imgs)
                img_paths.extend(personal_img_paths)
            model_results = {}
            for fr_name in all_fr_names:
                fr = FR(model_name=fr_name, device=device)
                model_results[fr_name] = get_negative_samples(face_db=face_db, fr=fr, device=device, query_portion=0.5)
                print(f"Model {fr_name} found {len(model_results[fr_name])} negative samples.")
            with open(negative_sample_res_path, 'w') as f:
                yaml.dump(model_results, f)
        else:
            with open(negative_sample_res_path, 'r') as f:
                model_results = yaml.safe_load(f)
                print(f"Loaded negative samples from {negative_sample_res_path}")
        diversities = {}
        highest_div_all, best_ens_all = -1, None
        for ensemble_size in ensemble_sizes:
            highest_div, best_ens = -1, None
            for ensemble in combinations(all_fr_names, ensemble_size):
                div = cal_focal_diversity(ensemble=list(ensemble), model_results=model_results)
                diversities["|".join(ensemble)] = float(div)
                if div > highest_div:
                    highest_div = div
                    best_ens = ensemble
            print(f"Best ensemble of size {ensemble_size} is {best_ens} with focal diversity {highest_div:.4f}")
            if highest_div > highest_div_all:
                highest_div_all = highest_div
                best_ens_all = best_ens
        with open(diversity_res_path, 'w') as f:
            yaml.dump(diversities, f)
        print(f"Overall best ensemble is {best_ens_all} with focal diversity {highest_div_all:.4f} and size {len(best_ens_all)}")
"""
(protego_plus) zlwang@kachow-g2:~/ProtegoPlus$ python3 focal_diversity.py 
Model ir50_softmax_casia found 1384 negative samples.
Model ir50_cosface_casia found 693 negative samples.
Model ir50_arcface_casia found 3460 negative samples.
Model mobilenet_arcface_casia found 1134 negative samples.
Model mobilefacenet_arcface_casia found 3872 negative samples.
Model ir18_adaface_webface found 871 negative samples.
Model ir50_adaface_ms1mv2 found 188 negative samples.
Model ir50_adaface_casia found 993 negative samples.
Model ir50_adaface_webface found 185 negative samples.
Model ir101_adaface_webface found 126 negative samples.
Model inception_facenet_vgg found 507 negative samples.
Model inception_facenet_casia found 1017 negative samples.
Model ir50_magface_ms1mv2 found 222 negative samples.
Model ir100_magface_ms1mv2 found 150 negative samples.
Best ensemble of size 2 is ('ir18_adaface_webface', 'inception_facenet_vgg') with focal diversity 0.5897
Best ensemble of size 3 is ('ir18_adaface_webface', 'inception_facenet_vgg', 'inception_facenet_casia') with focal diversity 0.5330
Best ensemble of size 4 is ('ir18_adaface_webface', 'inception_facenet_vgg', 'inception_facenet_casia', 'ir50_magface_ms1mv2') with focal diversity 0.5182
Best ensemble of size 5 is ('mobilenet_arcface_casia', 'ir18_adaface_webface', 'inception_facenet_vgg', 'inception_facenet_casia', 'ir50_magface_ms1mv2') with focal diversity 0.5099
Best ensemble of size 6 is ('ir50_arcface_casia', 'mobilenet_arcface_casia', 'ir18_adaface_webface', 'inception_facenet_vgg', 'inception_facenet_casia', 'ir50_magface_ms1mv2') with focal diversity 0.4958
Best ensemble of size 7 is ('ir50_cosface_casia', 'ir50_arcface_casia', 'mobilenet_arcface_casia', 'ir18_adaface_webface', 'inception_facenet_vgg', 'inception_facenet_casia', 'ir50_magface_ms1mv2') with focal diversity 0.4850
Best ensemble of size 8 is ('ir50_cosface_casia', 'ir50_arcface_casia', 'mobilenet_arcface_casia', 'ir18_adaface_webface', 'ir50_adaface_casia', 'inception_facenet_vgg', 'inception_facenet_casia', 'ir50_magface_ms1mv2') with focal diversity 0.4745
Best ensemble of size 9 is ('ir50_cosface_casia', 'ir50_arcface_casia', 'mobilenet_arcface_casia', 'mobilefacenet_arcface_casia', 'ir18_adaface_webface', 'ir101_adaface_webface', 'inception_facenet_vgg', 'inception_facenet_casia', 'ir50_magface_ms1mv2') with focal diversity 0.4665
Best ensemble of size 10 is ('ir50_cosface_casia', 'ir50_arcface_casia', 'mobilenet_arcface_casia', 'mobilefacenet_arcface_casia', 'ir18_adaface_webface', 'ir50_adaface_casia', 'ir101_adaface_webface', 'inception_facenet_vgg', 'inception_facenet_casia', 'ir50_magface_ms1mv2') with focal diversity 0.4610
Best ensemble of size 11 is ('ir50_softmax_casia', 'ir50_cosface_casia', 'ir50_arcface_casia', 'mobilenet_arcface_casia', 'mobilefacenet_arcface_casia', 'ir18_adaface_webface', 'ir50_adaface_casia', 'ir101_adaface_webface', 'inception_facenet_vgg', 'inception_facenet_casia', 'ir50_magface_ms1mv2') with focal diversity 0.4545
Best ensemble of size 12 is ('ir50_softmax_casia', 'ir50_cosface_casia', 'ir50_arcface_casia', 'mobilenet_arcface_casia', 'mobilefacenet_arcface_casia', 'ir18_adaface_webface', 'ir50_adaface_casia', 'ir101_adaface_webface', 'inception_facenet_vgg', 'inception_facenet_casia', 'ir50_magface_ms1mv2', 'ir100_magface_ms1mv2') with focal diversity 0.4479
Best ensemble of size 13 is ('ir50_softmax_casia', 'ir50_cosface_casia', 'ir50_arcface_casia', 'mobilenet_arcface_casia', 'mobilefacenet_arcface_casia', 'ir18_adaface_webface', 'ir50_adaface_casia', 'ir50_adaface_webface', 'ir101_adaface_webface', 'inception_facenet_vgg', 'inception_facenet_casia', 'ir50_magface_ms1mv2', 'ir100_magface_ms1mv2') with focal diversity 0.4396
Best ensemble of size 14 is ('ir50_softmax_casia', 'ir50_cosface_casia', 'ir50_arcface_casia', 'mobilenet_arcface_casia', 'mobilefacenet_arcface_casia', 'ir18_adaface_webface', 'ir50_adaface_ms1mv2', 'ir50_adaface_casia', 'ir50_adaface_webface', 'ir101_adaface_webface', 'inception_facenet_vgg', 'inception_facenet_casia', 'ir50_magface_ms1mv2', 'ir100_magface_ms1mv2') with focal diversity 0.4304
Overall best ensemble is ('ir18_adaface_webface', 'inception_facenet_vgg') with focal diversity 0.5897 and size 2
"""