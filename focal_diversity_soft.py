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


def compute_per_image_recall(
    face_db: Dict[str, Tuple[List[str], torch.Tensor]],
    fr: FR,
    device: torch.device,
    query_portion: float = 0.5,
    topk: int = 5,
) -> Dict[str, float]:
    """
    Compute per-image recall for a given FR model.

    Definition (aligned with prot_eval's retrieve usage):
    - For each query image, recall := (# of correct-identity images among Top-K retrievals) / K.

    Implementation details:
    - For each identity (name), split its images into queries (first portion) and gallery (remaining portion).
    - The database for retrieval is: all other identities + the gallery portion of the same identity.
    - We evaluate Top-K retrieval on the query features and record recall for each query image path.

    Returns:
    - Dict[img_path, recall ∈ [0,1]]. Only query images are keys.
    """
    # Extract features per identity, keep per-identity image paths to index queries accurately
    features_dict: Dict[str, torch.Tensor] = {}
    name_to_paths: Dict[str, List[str]] = {}
    for name, (imgs_paths, imgs) in face_db.items():
        features_dict[name] = fr(imgs.to(device)).cpu()
        name_to_paths[name] = list(imgs_paths)  # preserve order

    per_image_recall: Dict[str, float] = {}
    for name, features in features_dict.items():
        # Build retrieval database: all OTHER identities + gallery of the SAME identity
        db_parts, db_labels = [], []
        for n, f in features_dict.items():
            if n != name:
                db_parts.append(f)
                db_labels.extend([n] * f.shape[0])
        db = torch.cat(db_parts, dim=0).to(device)

        feature_nums = features.shape[0]
        query_nums = int(feature_nums * query_portion)
        queries = features[:query_nums].to(device)
        query_labels = [name] * query_nums

        # Append gallery portion of current identity into DB
        db = torch.cat([db, features[query_nums:].to(device)], dim=0)
        db_labels.extend([name] * (feature_nums - query_nums))

        # Retrieve with Top-K and record recall per query
        res = retrieve(
            db=db,
            db_labels=db_labels,
            queries=queries,
            query_labels=query_labels,
            dist_func='cosine',
            topk=feature_nums - query_nums
        )

        # Map each query index back to its image path
        query_img_paths = name_to_paths[name][:query_nums]
        for idx, recall in enumerate(res):
            per_image_recall[query_img_paths[idx]] = float(recall)

    return per_image_recall


def cal_focal_diversity_soft(
    ensemble: List[str],
    model_recalls: Dict[str, Dict[str, float]],
) -> float:
    """
    Soft focal diversity using per-image recall rates to measure disagreement.

    Notation
    - For model M and image x, let r_M(x) ∈ [0,1] be the Top-K recall for x.
    - Define negative degree n_M(x) := 1 - r_M(x) (higher => more negative/failure-like).

    Definition (soft analogue of the paper's focal diversity):
    - For each focal model F in ensemble T, compute a weighted agreement:
        λ_focal(T; F) := (Σ_x n_F(x) · avg_{M∈T\{F}} n_M(x)) / (Σ_x n_F(x)).
      If Σ_x n_F(x) = 0 (no negatives for F), skip F (no evidence).
    - The soft focal diversity is the average over focal members with evidence:
        d_focal_soft(T) := avg_{F∈T, Σ_x n_F(x)>0} [1 - λ_focal(T; F)].

    Intuition
    - When F has high negative degree on some images, we check whether others are also negative on those images.
      If others are also negative (high agreement), λ is large => diversity small; if others are strong, λ small => diversity large.
    """
    S = len(ensemble)
    if S <= 1:
        return 0.0

    total_div = 0.0
    eff_count = 0

    for focal in ensemble:
        focal_recalls = model_recalls.get(focal, {})
        if not focal_recalls:
            continue

        weight_sum = 0.0  # Σ_x n_F(x)
        agree_sum = 0.0   # Σ_x n_F(x) * avg_{others} n_M(x)

        for x, r_f in focal_recalls.items():
            n_f = 1.0 - float(r_f)
            if n_f <= 0.0:
                continue  # no weight for perfectly recalled samples

            # Compute average negative degree among the other models for this x
            n_sum = 0.0
            n_cnt = 0
            for other in ensemble:
                if other == focal:
                    continue
                r_o = model_recalls.get(other, {}).get(x, None)
                if r_o is None:
                    continue
                n_sum += (1.0 - float(r_o))
                n_cnt += 1
            if n_cnt == 0:
                continue  # no evidence from others for this sample

            n_others_avg = n_sum / n_cnt
            weight_sum += n_f
            agree_sum += n_f * n_others_avg

        if weight_sum == 0.0:
            # No informative negatives for this focal model; skip
            continue

        lambda_focal = agree_sum / weight_sum  # ∈ [0,1]
        total_div += (1.0 - lambda_focal)
        eff_count += 1

    if eff_count == 0:
        return 0.0
    d_focal = total_div / eff_count
    d_focal = max(0.0, min(1.0, d_focal))
    return d_focal


if __name__ == "__main__":
    # Configuration
    with torch.no_grad():
        face_db_base_path = "/home/zlwang/ProtegoPlus/face_db/face_scrub"
        device = torch.device("cuda:1")
        all_fr_names = [n for n in BASIC_POOL + SPECIAL_POOL if n not in VIT_FAMILY]
        ensemble_sizes = [i for i in range(2, len(all_fr_names) + 1)]
        per_image_recall_path = "/home/zlwang/ProtegoPlus/results/eval/per_image_recalls_soft_new.yaml"
        overwrite = True
        diversity_res_path = "/home/zlwang/ProtegoPlus/results/eval/focal_diversity_soft_new.yaml"
        topk = 5

        # Build face DB (images and paths)
        db_names = [
            os.path.join(face_db_base_path, n)
            for n in os.listdir(face_db_base_path)
            if not n.startswith((".", "_"))
        ]
        db_names.extend(
            [
                os.path.join(face_db_base_path, "_noise_db", n)
                for n in os.listdir(os.path.join(face_db_base_path, "_noise_db"))
                if not n.startswith((".", "_"))
            ]
        )

        face_db: Dict[str, Tuple[List[str], torch.Tensor]] = {}
        for name in db_names:
            imgs, personal_img_paths = load_imgs(
                base_dir=name,
                img_sz=224,
                usage_portion=1.0,
                drange=1,
                device=torch.device("cpu"),
                return_img_paths=True,
            )
            face_db[name] = (personal_img_paths, imgs)

        # Compute or load per-image recalls for each FR
        if overwrite:
            model_recalls: Dict[str, Dict[str, float]] = {}
            for fr_name in all_fr_names:
                fr = FR(model_name=fr_name, device=device)
                per_img_recall = compute_per_image_recall(
                    face_db=face_db,
                    fr=fr,
                    device=device,
                    query_portion=0.5,
                    topk=topk,
                )
                model_recalls[fr_name] = per_img_recall
                print(
                    f"Model {fr_name} computed {len(per_img_recall)} per-image recalls (queries only, topk={topk})."
                )
            with open(per_image_recall_path, "w") as f:
                yaml.dump(model_recalls, f)
        else:
            with open(per_image_recall_path, "r") as f:
                model_recalls = yaml.safe_load(f)
                print(f"Loaded per-image recalls from {per_image_recall_path}")

        # Enumerate ensembles and compute soft focal diversity
        diversities: Dict[str, float] = {}
        highest_div_all, best_ens_all = -1.0, None
        for ensemble_size in ensemble_sizes:
            highest_div, best_ens = -1.0, None
            for ensemble in combinations(all_fr_names, ensemble_size):
                div = cal_focal_diversity_soft(ensemble=list(ensemble), model_recalls=model_recalls)
                diversities["|".join(ensemble)] = float(div)
                if div > highest_div:
                    highest_div = div
                    best_ens = ensemble
            print(
                f"[Soft] Best ensemble of size {ensemble_size} is {best_ens} with focal diversity {highest_div:.4f}"
            )
            if highest_div > highest_div_all:
                highest_div_all = highest_div
                best_ens_all = best_ens

        with open(diversity_res_path, "w") as f:
            yaml.dump(diversities, f)
        print(
            f"[Soft] Overall best ensemble is {best_ens_all} with focal diversity {highest_div_all:.4f} and size {len(best_ens_all)}"
        )
