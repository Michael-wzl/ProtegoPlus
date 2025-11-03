from itertools import combinations
from typing import Dict, List, Tuple, Set, Union, Any, Optional
import re
from collections import Counter

import torch
import numpy as np
import tqdm

from .utils import retrieve, build_facedb

def get_majority_class(labels: List[str]) -> str:
    """
    Get the majority class from the list of labels. Ordered by frequency, first occurrence, then lexicographically.

    Args:
        labels (List[str]): The list of labels.
    
    Returns:
        str: The majority class label.
    """
    label_counts = Counter(labels)
    max_count = max(label_counts.values())
    candidates = [label for label, count in label_counts.items() if count == max_count]
    # Tie-break by first occurrence in the list
    for label in labels:
        if label in candidates:
            return label
    # Fallback (should not reach here)
    return sorted(candidates)[0]

def get_retrieval_results(
    features: torch.Tensor, 
    labels: List[str], 
    face_db: Dict[str, Tuple[int, int]], 
    query_id_portion: float = 1., 
    query_img_portion: float = 0.5, 
    put_back: bool = False, 
    device: torch.device = torch.device('cpu')
) -> Dict[int, Tuple[float, List[int], List[str], str]]:
    """
    Perform retrieval on the given features and labels.

    Args:
        features (torch.Tensor): The feature tensor of shape (N, D).
        labels (List[str]): The list of labels corresponding to the features.
        face_db (Dict[str, Tuple[int, int]]): The face database mapping names to (start_idx, end_idx). The end_idx is exclusive, aka the start_idx of the next identity.
        query_id_portion (float): The portion of IDs to use as queries.
        query_img_portion (float): The portion of images per ID to use as queries.
        put_back (bool): Whether to put back the other IDs' query images into the database when retrieving for an ID. 
        device (torch.device): The device to perform computations on.

    Returns:
        Dict[int, Tuple[float, List[int], List[str], str]]: A mapping from query feature index to a tuple of
            (recall, retrieved_global_indices, retrieved_global_labels, gt_label). All indexes are in the
            global feature space and decreasing in feature similarity.
    """
    id_num = len(face_db)
    results = {}
    target_id_count = max(1, int(id_num * float(query_id_portion)))
    if put_back:
        for name_idx, (name, (start_idx, end_idx)) in enumerate(face_db.items()):
            if name_idx >= target_id_count:
                break
            query_end_idx = start_idx + int((end_idx - start_idx) * query_img_portion)
            query_features = features[start_idx:query_end_idx].to(device)
            db_features = torch.cat([features[:start_idx], features[query_end_idx:]], dim=0).to(device)
            query_labels = labels[start_idx:query_end_idx]
            db_labels = labels[:start_idx] + labels[query_end_idx:]
            recalls, retrieved_idxs = retrieve(
                db=db_features, 
                db_labels=db_labels, 
                queries=query_features,
                query_labels=query_labels, 
                dist_func='cosine', 
                topk=end_idx - query_end_idx, 
                sorted_retrieval=True,
                return_retrieved_idxs=True
            )
            shift = (query_end_idx - start_idx)
            for i, retrieved_idx in enumerate(retrieved_idxs):
                global_retrieved_idxs = []
                global_retrieved_labels = []
                for ind in retrieved_idx:
                    if ind < start_idx:
                        global_retrieved_idxs.append(ind)
                    else:
                        global_retrieved_idxs.append(ind + shift)
                global_retrieved_labels = [labels[idx] for idx in global_retrieved_idxs]
                # ground-truth label of this query image
                gt_label = query_labels[i]
                results[start_idx + i] = (recalls[i], global_retrieved_idxs, global_retrieved_labels, gt_label)
    else:
        db_features, query_features = [], {}
        db_labels, query_labels = [], {}
        for name_idx, (name, (start_idx, end_idx)) in enumerate(face_db.items()):
            if name_idx < target_id_count:
                query_end_idx = start_idx + int((end_idx - start_idx) * query_img_portion)
                db_features.append(features[query_end_idx:end_idx])
                query_features[name] = (features[start_idx:query_end_idx], end_idx - query_end_idx)
                db_labels.extend(labels[query_end_idx:end_idx])
                query_labels[name] = labels[start_idx:query_end_idx]
            else:
                db_features.append(features[start_idx:end_idx])
                db_labels.extend(labels[start_idx:end_idx])
        db_features = torch.cat(db_features, dim=0).to(device)
        start_idx = 0
        for name, (query_feats, topk) in query_features.items():
            query_lbls = query_labels[name]
            recalls, retrieved_idxs = retrieve(
                db=db_features, 
                db_labels=db_labels, 
                queries=query_feats,
                query_labels=query_lbls,
                dist_func='cosine', 
                topk=topk,
                sorted_retrieval=True,
                return_retrieved_idxs=True
            )
            for i, retrieved_idx in enumerate(retrieved_idxs):
                global_retrieved_idxs = retrieved_idx
                global_retrieved_labels = [db_labels[idx] for idx in global_retrieved_idxs]
                # ground-truth label of this query image
                gt_label = query_lbls[i]
                results[start_idx + i] = (recalls[i], global_retrieved_idxs, global_retrieved_labels, gt_label)
            start_idx += query_feats.shape[0]
    return results

def get_focal_diversity(ensemble_retrieval_results: List[Dict[int, Tuple[float, List[int], List[str], str]]], definition: str) -> float:
    """
    Compute the focal diversity of the ensemble retrieval results.

    Args:
        ensemble_retrieval_results (List[Dict[int, Tuple[float, List[int], List[str], str]]]): The ensemble retrieval results.
    definition (str): The definition of focal diversity to use. Supports 'jaccard_absrecall', 'top{k}classification', 'soft_classification'

    Returns:
        float: The focal diversity (or ensemble quality) score.
    """
    if definition == 'performance_only':
        avg_recall = []
        for query_idx in ensemble_retrieval_results[0].keys():
            recalls = []
            for model_results in ensemble_retrieval_results:
                recall, _, _, _ = model_results[query_idx]
                recalls.append(float(recall))
            avg_recall.append(np.mean(recalls))
        return float(np.mean(avg_recall))
    elif definition == 'intersectional_size':
        agreements = []
        for i, query_idx in enumerate(ensemble_retrieval_results[0].keys()):
            retrieved_sets = []
            for model_results in ensemble_retrieval_results:
                _, retrieved_idxs, _, _ = model_results[query_idx]
                #if i == 0:
                #    print(retrieved_idxs)
                retrieved_sets.append(set(retrieved_idxs))
            total_retrieval = len(retrieved_sets[0])
            #print(total_retrieval)
            intersection_size = len(set.intersection(*retrieved_sets))
            agreements.append(intersection_size / total_retrieval if total_retrieval > 0 else 0.0)
        return float(1 - np.mean(agreements))
    elif definition == 'intersectional_focal':
        neg_lambdas = []
        for focal_model in range(len(ensemble_retrieval_results)):
            agreements = []
            for query_idx in ensemble_retrieval_results[focal_model].keys():
                _, retrieved_idxs_focal, _, _ = ensemble_retrieval_results[focal_model][query_idx]
                per_img_agreements = []
                for other_model in range(len(ensemble_retrieval_results)):
                    if other_model == focal_model:
                        continue
                    _, retrieved_idxs_other, _, _ = ensemble_retrieval_results[other_model][query_idx]
                    intersec_size = len(set(retrieved_idxs_focal).intersection(set(retrieved_idxs_other)))
                    total_size = len(retrieved_idxs_focal)
                    per_img_agreements.append(intersec_size / total_size if total_size > 0 else 0.0)
                agreements.append(np.mean(per_img_agreements))
            if not agreements:
                continue
            lambda_focal = np.mean(agreements)
            neg_lambdas.append(1 - lambda_focal)
        if not neg_lambdas:
            return 0.0
        d_focal = np.mean(neg_lambdas)
        return float(d_focal)
    elif definition == 'jaccard_absrecall':
        neg_lambdas = []
        for focal_model in range(len(ensemble_retrieval_results)):
            neg_weight_sum = 0.0
            agreement_sum = 0.0
            for query_idx in ensemble_retrieval_results[focal_model].keys():
                recall_focal, _, retrieved_labels_focal, _ = ensemble_retrieval_results[focal_model][query_idx]
                n_focal = 1.0 - float(recall_focal)
                if n_focal <= 0.0:
                    continue
                perimg_agree_sum = 0.0
                perimg_agree_cnt = 0
                for other_model in range(len(ensemble_retrieval_results)):
                    if other_model == focal_model:
                        continue
                    recall_other, _, retrieved_labels_other, _ = ensemble_retrieval_results[other_model][query_idx]
                    n_other = 1.0 - float(recall_other)
                    agreement_score = n_other * np.mean([1.0 if label == retrieved_labels_other[i] else 0.0 for i, label in enumerate(retrieved_labels_focal)])
                    #jaccard_index = len(set(retrieved_labels_focal).intersection(set(retrieved_labels_other))) / len(set(retrieved_labels_focal).union(set(retrieved_labels_other)))
                    #agreement_score = n_other * jaccard_index
                    perimg_agree_sum += agreement_score
                    perimg_agree_cnt += 1
                neg_weight_sum += n_focal
                agreement_sum += (n_focal * (perimg_agree_sum / max(1, perimg_agree_cnt)))
            if neg_weight_sum == 0.0:
                continue
            lambda_focal = agreement_sum / neg_weight_sum
            neg_lambdas.append(1 - lambda_focal)
        if not neg_lambdas:
            return 0.0
        d_focal = np.mean(neg_lambdas)
        return float(max(0.0, min(1.0, d_focal)))
        """all_disagreement_scores = []
        model_pair_combs = list(combinations(range(len(ensemble_retrieval_results)), 2))
        for model_a_idx, model_b_idx in model_pair_combs:
            model_a_results = ensemble_retrieval_results[model_a_idx]
            model_b_results = ensemble_retrieval_results[model_b_idx]
            disagreement_scores = []
            for query_idx in model_a_results.keys():
                recall_a, retrieved_a, _, _ = model_a_results[query_idx]
                recall_b, retrieved_b, _, _ = model_b_results[query_idx]
                set_a = set(retrieved_a)
                set_b = set(retrieved_b)
                intersection_size = len(set_a.intersection(set_b))
                union_size = len(set_a.union(set_b))
                jaccard_index = intersection_size / union_size if union_size > 0 else 0.0
                disagreement_score = (1.0 - jaccard_index + abs(recall_a - recall_b)) / 2.0
                disagreement_scores.append(disagreement_score)
            avg_disagreement_score = np.mean(disagreement_scores)
            all_disagreement_scores.append(avg_disagreement_score)
        # Return a native Python float to avoid serialization issues downstream
        return float(np.mean(all_disagreement_scores))"""
    elif 'top' in definition and 'classification' in definition:
        m = re.search(r"top(\d+)", definition)
        k = int(m.group(1)) if m else 1
        S = len(ensemble_retrieval_results)
        # 1) Pre-compute negative sets (queries where GT not in top-k retrieved labels) for each model
        negative_sets: List[Set[int]] = []
        for model_results in ensemble_retrieval_results:
            negs: Set[int] = set()
            for q_idx, tup in model_results.items():
                # tuple layout: (recall, retrieved_idxs, retrieved_labels, gt_label)
                _, _, retrieved_labels, gt_label = tup
                topk_labels = retrieved_labels[:k]
                majority_label = get_majority_class(topk_labels)
                if majority_label != gt_label:
                    negs.add(q_idx)
            negative_sets.append(negs)
        # 2) For each focal model, estimate lambda_focal as average agreement of failures on its negatives
        total_div = 0.0
        eff_count = 0
        denom_models = max(1, S - 1)
        for focal_idx in range(S):
            focal_negs = negative_sets[focal_idx]
            if not focal_negs:
                # Skip focal members with no negatives (no evidence)
                continue
            agree_sum = 0.0
            for q in focal_negs:
                also_fail = 0
                for other_idx in range(S):
                    if other_idx == focal_idx:
                        continue
                    if q in negative_sets[other_idx]:
                        also_fail += 1
                agree_sum += (also_fail / denom_models)
            lambda_focal = agree_sum / max(1, len(focal_negs))
            total_div += (1.0 - lambda_focal)
            eff_count += 1
        if eff_count == 0:
            return 0.0
        d_focal = total_div / eff_count
        return float(max(0.0, min(1.0, d_focal)))
    elif definition == 'soft_classification':
        # Soft focal diversity based on per-query recall rates r_M(q) in [0,1]
        # n_M(q) := 1 - r_M(q). For each focal model F:
        #   lambda_F = (Σ_q n_F(q) * avg_{M≠F} n_M(q)) / (Σ_q n_F(q)) over queries with n_F(q) > 0
        # d_focal_soft = avg_F [1 - lambda_F] over focal members with evidence
        S = len(ensemble_retrieval_results)
        recall_maps: List[Dict[int, float]] = []
        # 1) Build recall maps for each model
        for model_results in ensemble_retrieval_results:
            rmap: Dict[int, float] = {}
            for q_idx, tup in model_results.items():
                # tuple layout: (recall, retrieved_idxs, retrieved_labels, gt_label)
                r = float(tup[0])
                rmap[q_idx] = r
            recall_maps.append(rmap)
        # 2) For each focal model, estimate lambda_focal
        total_div = 0.0
        eff_count = 0
        for f_idx in range(S):
            focal_map = recall_maps[f_idx]
            weight_sum = 0.0  # Σ_q n_F(q)
            agree_sum = 0.0   # Σ_q n_F(q) * avg_{others} n_M(q)
            for q_idx, r_f in focal_map.items():
                n_f = 1.0 - float(r_f)
                if n_f <= 0.0:
                    continue
                n_sum = 0.0
                n_cnt = 0
                for o_idx in range(S):
                    if o_idx == f_idx:
                        continue
                    r_o = recall_maps[o_idx][q_idx]
                    n_sum += (1.0 - float(r_o))
                    n_cnt += 1
                n_others_avg = n_sum / n_cnt
                weight_sum += n_f
                agree_sum += n_f * n_others_avg
            if weight_sum == 0.0:
                # No informative negatives for this focal model; skip
                continue
            lambda_focal = agree_sum / weight_sum
            total_div += (1.0 - lambda_focal)
            eff_count += 1
        if eff_count == 0:
            return 0.0
        d_focal = total_div / eff_count
        return float(max(0.0, min(1.0, d_focal)))
    else:
        raise ValueError(f"Unsupported focal diversity definition: {definition}.")

def remove_common_retrievals(retrieval_results: List[Dict[int, Tuple[float, List[int], List[str], str]]], class_based: bool, topk: int = 5) -> List[Dict[int, Tuple[float, List[int], List[str], str]]]:
    """
    Remove query images that all models retrieve the same set of images or classified as the same class. 

    Args:
        retrieval_results (List[Dict[int, Tuple[float, List[int], List[str], str]]]): The retrieval results for each model.
        class_based (bool): Whether to consider class-based retrievals (based on labels) or index-based retrievals.
        topk (Optional[int]): If class_based is True, consider only the top-k retrieved labels for classification comparison.

    Returns:
        List[Dict[int, Tuple[float, List[int], List[str], str]]]: The filtered retrieval results.
    """
    adjusted_results: List[Dict[int, Tuple[float, List[int], List[str], str]]] = [{} for _ in retrieval_results]
    for query, retrieval_result in retrieval_results[0].items():
        is_common = True
        if class_based:
            _, _, retrieved_labels_0, _ = retrieval_result
            majority_label_0 = get_majority_class(retrieved_labels_0[:topk])
            for model_results in retrieval_results[1:]:
                _, _, retrieved_labels_i, _ = model_results[query]
                majority_label_i = get_majority_class(retrieved_labels_i[:topk])
                if majority_label_0 != majority_label_i:
                    is_common = False
                    break
        else:
            _, retrieved_idxs_0, _, _ = retrieval_result
            for model_results in retrieval_results[1:]:
                _, retrieved_idxs_i, _, _ = model_results[query]
                if set(retrieved_idxs_0) != set(retrieved_idxs_i):
                    is_common = False
                    break
        if not is_common:
            for model_idx, model_results in enumerate(retrieval_results):
                adjusted_results[model_idx][query] = model_results[query]
    return adjusted_results

def get_focal_diversities(
    model_pool: List[str], 
    ensemble_size: int, 
    feature_base_paths: List[str], 
    query_id_portion: float,
    query_img_portion: float,
    standardize_size: int,
    allow_dup: bool, 
    definition: str, 
    remove_common: bool, 
    put_back: bool, 
    device: torch.device, 
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compute the focal diversities for all combinations of models in the model pool.

    Args:
        model_pool (List[str]): The list of model names to consider.
        ensemble_size (int): The size of the ensemble.
        feature_base_paths (List[str]): The list of feature base paths.
        query_id_portion (float): The portion of IDs to use as queries.
        query_img_portion (float): The portion of images per ID to use as queries.
        standardize_size (int): The size to standardize the number of images per identity across databases. If -1, no standardization.
        allow_dup (bool): Whether to allow duplication of images when standardizing size.
        definition (str): The definition of focal diversity to use.
        remove_common (bool): Whether to remove query images that all models retrieve the same set of images.
        put_back (bool): Whether to put back the other IDs' query images into the database when retrieving for an ID. 
        device (torch.device): The device to perform computations on.
        verbose (bool): Whether to print verbose output.

    Returns:
        Dict[str, Any]: Consisting of:
         - 'retrieval_results' (Dict[str, Dict[int, Tuple[float, List[int]]]]): The retrieval results for each model.
         - 'focal_diversities' (Dict[List[str], float]): The focal diversities for each model combination.
    """
    retrieval_results = {}
    if verbose:
        pbar = tqdm.tqdm(enumerate(model_pool), total=len(model_pool), desc="Calculating retrieval results")
    else:
        pbar = enumerate(model_pool)
    for model_idx, model_name in pbar:
        if verbose:
            pbar.set_description(f"Calculating retrieval results for {model_name}")
        features, labels = [], []
        face_db = {}
        start_idx = 0
        # Preserve insertion order of identities explicitly to avoid relying on dict order
        identity_names: List[str] = []
        for feature_base_path in feature_base_paths:
            for name, feature_tensor in build_facedb(db_path=feature_base_path, fr=model_name, device=device).items():
                feature_num = feature_tensor.shape[0]
                if name not in face_db:
                    end_idx = start_idx + feature_num
                    face_db[name] = (start_idx, end_idx)
                    identity_names.append(name)
                    features.append(feature_tensor)
                    labels.extend([name] * feature_num)
                    start_idx = end_idx
                else:
                    if verbose:
                        print(f"[WARN] Duplicate identity {name} found for model {model_name} in path {feature_base_path}. Skipping.")
                    continue
        if standardize_size > 0:
            new_features, new_labels, new_face_db = [], [], {}
            curr_start_idx = 0
            # Iterate using the explicit identity order captured above to guarantee alignment
            for name_idx, name in enumerate(identity_names):
                start_idx_id, end_idx_id = face_db[name]
                current_size = end_idx_id - start_idx_id
                # features[name_idx] corresponds to the same insertion order as identity_names
                feature_tensor = features[name_idx]
                # Labels for this identity are all the same; derive directly to avoid any slice mismatch risk
                label_segment = [name] * current_size
                if current_size < standardize_size:
                    if (standardize_size - current_size) / standardize_size > 0.1 or not allow_dup:
                        if verbose and model_idx == 0:
                            print(f"[WARN] Identity {name} has only {current_size} images, cannot standardize to {standardize_size}. Skipping.")
                        continue
                    else:
                        if verbose and model_idx == 0:
                            print(f"[INFO] Identity {name} has only {current_size} images, duplicating to reach standardize size {standardize_size}.")
                        # Duplicate samples to reach standardize_size
                        patch_size = standardize_size - current_size
                        dup_features = torch.cat([feature_tensor, feature_tensor[:patch_size]], dim=0)
                        dup_labels = label_segment + label_segment[:patch_size]
                        new_end_idx = curr_start_idx + standardize_size
                        new_face_db[name] = (curr_start_idx, new_end_idx)
                        new_features.append(dup_features)
                        new_labels.extend(dup_labels)
                else:
                    assert current_size >= standardize_size, f"Identity {name} has only {current_size} images, cannot standardize to {standardize_size}."
                    selected_features = feature_tensor[:standardize_size]
                    selected_labels = label_segment[:standardize_size]
                    new_end_idx = curr_start_idx + standardize_size
                    new_face_db[name] = (curr_start_idx, new_end_idx)
                    new_features.append(selected_features)
                    new_labels.extend(selected_labels)
                curr_start_idx = new_end_idx
            features = new_features
            labels = new_labels
            face_db = new_face_db
        # Concatenate to global feature matrix and sanity-check index consistency
        features = torch.cat(features, dim=0)
        # Optional consistency checks (kept lightweight; assert can be disabled with -O)
        try:
            total_len = features.shape[0]
            assert len(labels) == total_len, f"Features/labels length mismatch: {total_len} vs {len(labels)}"
            # Verify face_db spans the full range without gaps/overlaps
            if face_db:
                spans = sorted(face_db.values(), key=lambda x: x[0])
                last_end = 0
                for s, e in spans:
                    assert s == last_end, f"Non-contiguous index span: expected start {last_end}, got {s}"
                    assert 0 <= s < e <= total_len, f"Span out of bounds: ({s},{e}) not within [0,{total_len}]"
                    last_end = e
                assert last_end == total_len, f"Final end index {last_end} != total length {total_len}"
        except AssertionError as ae:
            # Surface a clearer message to aid debugging while keeping behavior unchanged
            raise AssertionError(f"[standardize_size] index consistency check failed for model '{model_name}': {ae}")
        retrieval_results[model_name] = get_retrieval_results(
            features=features, 
            labels=labels, 
            face_db=face_db, 
            query_id_portion=query_id_portion, 
            query_img_portion=query_img_portion, 
            put_back=put_back,
            device=device
        )
    if remove_common:
        if verbose:
            print("Removing common retrievals across models...")
        retrieval_results_list = [retrieval_results[model_name] for model_name in model_pool]
        classed_based = ('classification' in definition and 'top' in definition)
        m = re.search(r"top(\d+)", definition)
        k = int(m.group(1)) if m else 1
        adjusted_results_list = remove_common_retrievals(
            retrieval_results=retrieval_results_list, 
            class_based=classed_based,
            topk=k
        )
        for model_idx, model_name in enumerate(model_pool):
            retrieval_results[model_name] = adjusted_results_list[model_idx]
    elif verbose:
        # Optional sanity check: ensure query key sets align across models (important for put_back)
        key_ref: Optional[Set[int]] = None
        for model_name in model_pool:
            ks = set(retrieval_results[model_name].keys())
            if key_ref is None:
                key_ref = ks
            else:
                if ks != key_ref:
                    inter = len(ks & key_ref)
                    print(f"[WARN] Query key set mismatch for model {model_name}: |A|={len(key_ref)}, |B|={len(ks)}, |A∩B|={inter}")
    focal_diversities = {}
    model_combs = list(combinations(model_pool, ensemble_size))
    if verbose:
        print(f"Calculating focal diversities for {len(model_combs)} model combinations of size {ensemble_size}")

    for model_name, res in retrieval_results.items():
        print(f"{model_name}: {res[0]}")
    
    for model_comb in model_combs:
        ensemble_retrieval_results = [retrieval_results[model_name] for model_name in model_comb]
        focal_diversity = get_focal_diversity(
            ensemble_retrieval_results=ensemble_retrieval_results, 
            definition=definition
        )
        # Store as native Python float for better interoperability
        focal_diversities[model_comb] = float(focal_diversity)
    return {'retrieval_results': retrieval_results, 'focal_diversities': focal_diversities}


"""
elif definition in ('soft_oracle', 'soft_oracle_gain', 'failure_correlation', 'failure_corr', 'dar'):
# New recall-correlated metrics
# Build recall maps r_M(q) in [0,1]
S = len(ensemble_retrieval_results)
recall_maps: List[Dict[int, float]] = []
for model_results in ensemble_retrieval_results:
    rmap: Dict[int, float] = {}
    for q_idx, tup in model_results.items():
        rmap[q_idx] = float(tup[0])
    recall_maps.append(rmap)

# Common set of queries (assume aligned; intersect for safety)
common_qs: Set[int] = set(recall_maps[0].keys())
for rm in recall_maps[1:]:
    common_qs &= set(rm.keys())
if not common_qs:
    return 0.0

eps = 1e-8

# 1) Soft Oracle Recall (SOR): expected success of ensemble under independence
#    SOR(q) = 1 - Π_M (1 - r_M(q)), SOR = avg_q SOR(q)
sor_vals = []
r_avg_vals = []
# Prepare failure matrices for correlation metric
failure_series: List[List[float]] = [[] for _ in range(S)]
for q in common_qs:
    one_minus = 1.0
    r_sum = 0.0
    for m in range(S):
        r = float(recall_maps[m][q])
        r = max(0.0, min(1.0, r))
        r_sum += r
        one_minus *= (1.0 - r)
        failure_series[m].append(1.0 - r)
    sor_q = 1.0 - one_minus
    sor_vals.append(sor_q)
    r_avg_vals.append(r_sum / S)

SOR = float(np.mean(sor_vals))
R_AVG = float(np.mean(r_avg_vals))

if definition == 'soft_oracle':
    return float(max(0.0, min(1.0, SOR)))

# 2) Soft Complementarity Gain (SCG): normalized gain over average model
#    SCG = (SOR - R_AVG) / (1 - R_AVG)
if definition == 'soft_oracle_gain':
    num = max(0.0, SOR - R_AVG)
    den = max(eps, 1.0 - R_AVG)
    scg = num / den
    return float(max(0.0, min(1.0, scg)))

# 3) Failure Correlation Index (FCI): 0.5*(1 - corr_avg) in [0,1]; higher is more diverse
#    Pairwise Pearson correlation across queries of failure series n_M(q) = 1 - r_M(q)
#    Skip pairs with zero variance
# Compute pairwise correlations
pair_corrs = []
for i in range(S):
    x = np.array(failure_series[i], dtype=np.float64)
    x_mean = x.mean()
    x_var = x.var()
    if x_var <= eps:
        continue
    x_std = np.sqrt(x_var)
    for j in range(i + 1, S):
        y = np.array(failure_series[j], dtype=np.float64)
        y_var = y.var()
        if y_var <= eps:
            continue
        y_std = np.sqrt(y_var)
        cov = float(np.mean((x - x_mean) * (y - y.mean())))
        corr = cov / (x_std * y_std)
        # Clip to valid range
        corr = max(-1.0, min(1.0, corr))
        pair_corrs.append(corr)
if not pair_corrs:
    corr_avg = 1.0  # fully correlated or degenerate -> zero diversity
else:
    corr_avg = float(np.mean(pair_corrs))
fci = 0.5 * (1.0 - corr_avg)
fci = float(max(0.0, min(1.0, fci)))

if definition in ('failure_correlation', 'failure_corr'):
    return fci

# 4) Diversity-Adjusted Recall (DAR): blends soft-oracle performance with diversity
#    DAR = SOR * (0.5 + 0.5*FCI), stays in [0,1]
if definition == 'dar':
    dar = SOR * (0.5 + 0.5 * fci)
    return float(max(0.0, min(1.0, dar)))
"""