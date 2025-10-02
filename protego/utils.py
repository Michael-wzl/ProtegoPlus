import os
from typing import Dict, List, Tuple, Any, Optional, Union
import gc
import sys
import filelock

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import PIL
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split as train_val_split
from pytorch_msssim import ssim as calc_ssim

from .FaceDetection import FD
from .FacialRecognition import FR
from .compression import compress

def crop_face(img: torch.Tensor, 
            detector: FD, 
            conf_thresh: float = 0.8, 
            min_h: int = 20, 
            min_w: int = 20, 
            crop_loosen: float = 1.0,
            verbose: bool = True) -> Tuple[Optional[torch.Tensor], Optional[Tuple[int, int, int, int]]]:
    """
    Crop the face from the image using MTCNN.

    Args:
        img (torch.Tensor): FloatTensor. Range [0, 255], RGB, [3, H, W]
        conf_thresh (float): Confidence threshold for face detection. Not passed to the detection model, used only in post-processing.
        min_h (int): Minimum height of the detected face.
        min_w (int): Minimum width of the detected face.
        crop_loosen (float): Factor to loosen the crop around the detected face. 1.0 means no change.
        verbose (bool): Whether to print warnings.

    Returns:
        Tuple[torch.Tensor, Tuple[int, int, int, int]]: The cropped face (Range [0, 255], RGB, [3, H, W]) and its position in the original image.
    """
    det_res = detector(img.float())
    if det_res is None:
        return None, None
    bboxs = [list(det_res[i][:5]) for i in range(len(det_res))]
    height, width = img.shape[1:]
    cropped_face = None
    position = None
    largest_area = 0
    for bbox in bboxs:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        conf = bbox[4] 
        if conf < conf_thresh:  # Confidence threshold
            if verbose:
                print(f"Discarding face with low confidence: {conf:.2f}")
            continue
        face_width = x2 - x1
        face_height = y2 - y1
        if face_height < min_h or face_width < min_w:
            if verbose:
                print(f"Discarding face with too small dimensions: width={face_width}, height={face_height}")
            continue
        face_area = face_width * face_height
        if face_area < largest_area:
            continue
        largest_area = face_area

        square_size = max(face_width, face_height)
        if crop_loosen != 1.0:
            square_size = int(square_size * crop_loosen)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        full_size = min(square_size, width, height)
        if full_size == width:
            if verbose:
                print(f"Warning: Original Image too narrow, cannot crop face properly.")
            square_x1, square_x2 = 0, width
            half_size = width // 2
            square_y1 = center_y - half_size
            square_y2 = center_y + half_size
        elif full_size == height:
            if verbose:
                print(f"Warning: Original Image too short, cannot crop face properly.")
            square_y1, square_y2 = 0, height
            half_size = height // 2
            square_x1 = center_x - half_size
            square_x2 = center_x + half_size
        elif full_size == square_size:
            half_size = full_size // 2
            square_x1 = center_x - half_size
            square_y1 = center_y - half_size
            square_x2 = center_x + half_size
            square_y2 = center_y + half_size

        cropped_face = img[:, square_y1:square_y2, square_x1:square_x2].clone()
        position = (square_x1, square_y1, square_x2, square_y2)

    return cropped_face, position

def complete_del() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps"):
        is_available = getattr(torch.mps, "is_available", None)
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if callable(is_available) and callable(empty_cache):
            try:
                if is_available():
                    empty_cache()
            except AttributeError:
                # Older PyTorch builds may expose torch.backends.mps without torch.mps helpers
                pass

def img2tensor(img: np.ndarray, drange: int = 1, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Convert a numpy image to a tensor.

    Args:
        img (np.ndarray): The numpy image to convert. Shape: (H, W, 3), dtype: uint8, range [0, 255].
        drange (int): The range of the image. Default is 1. Either 1 or 255.
        device (torch.device): The device to store the tensor. Default is CPU.

    Returns:
        torch.Tensor: The converted tensor. Shape: (1, 3, H, W), dtype: float32., range [0, 1].
    """
    img_tensor = torch.tensor(img, dtype=torch.float32, device=device).to(device)
    img_tensor = img_tensor.permute(2, 0, 1)
    if drange == 1:
        img_tensor /= 255. 
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def load_mask(mask_path: str, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Load a universal mask from a .npy file.

    Args:
        mask_path (str): The path to the .npy file containing the mask.
        device (torch.device): The device to store the tensor. Default is CPU.

    Returns:
        torch.Tensor: The loaded mask tensor. Shape: (1, 3, H, W), dtype: float32., range [0, 1].
    """
    return torch.tensor(np.load(mask_path, allow_pickle=True)[0], dtype=torch.float32, device=device).to(device).unsqueeze(0)

def load_imgs(base_dir: str = None, img_paths: List[str] = None, img_sz: int = 224, usage_portion: float = 1.0, drange: int = 1, device: torch.device = torch.device('cpu')) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Read images from a directory and convert them to tensors.

    Args:
        base_dir (str): The directory containing the images.
        img_paths (List[str]): The list of image paths. If provided, base_dir will be ignored.
        img_sz (int): The size of the image. If set to -1, it will not be resized. 
        usage_portion (float): The portion of the images to use.
        drange (int): The range of the image. Default is 1. Either 1 or 255.
        device (torch.device): The device to store the tensor. Default is CPU.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: The tensor of the images. Shape [B, 3, img_sz, img_sz]. RGB, range [0, 1], dtype: float32.
        A list of tensors if img_sz is -1.
    """
    img_tensors = []
    if base_dir is not None and img_paths is None:
        img_num = len(os.listdir(base_dir))
        imgs_names = sorted([os.path.join(base_dir, name) for name in os.listdir(base_dir) if name.endswith(('.jpg', '.png', '.jpeg', '.bmp')) and not (name.startswith('.') or name.startswith('_'))])
    elif img_paths is not None and base_dir is None:
        img_num = len(img_paths)
        imgs_names = img_paths
    else:
        raise ValueError("Either base_dir or img_paths should be provided, but not both or neither.")
    for img_path in imgs_names:
        img = PIL.Image.open(img_path).convert('RGB')
        img = np.array(img)
        if img_sz != -1:
            img = cv2.resize(img, (img_sz, img_sz), interpolation=cv2.INTER_LANCZOS4 if img.shape[0] < img_sz else cv2.INTER_AREA)
        img_tensors.append(img2tensor(img, drange=drange, device=device))
        if len(img_tensors) >= int(img_num * usage_portion):
            break
    return torch.cat(img_tensors, dim=0) if img_sz != -1 else img_tensors

def build_facedb(db_path: str, fr_name: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Build a database of features from the images in the given directory. Folders starting with '.' or '_' will be ignored.

    Args:
        db_path (str): The path to the database directory.
        fr_name (str): The name of the facial recognition model to use.
        device (torch.device): The device to use for computation.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the features for each person in the database.
    """
    db = {}
    for name in sorted(os.listdir(db_path)):
        if name.startswith('.') or name.startswith('_'):
            continue
        personal_path = os.path.join(db_path, name)
        features = torch.load(os.path.join(personal_path, f'{fr_name}.pt'))
        db[name] = features.to(device)
    return db

def build_compressed_face_db(db_path: str, fr: FR, device: torch.device, compression_methods: List[str], compression_cfgs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Build a database of features from the compressed images in the given directory.

    Args:
        db_path (str): The path to the database directory.
        fr_name (str): The name of the facial recognition model to use.
        device (torch.device): The device to use for computation.
        compression_methods (List[str]): The list of compression methods to use.
        compression_cfgs (Dict[str, Dict[str, Any]]): The configurations for each compression method.

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: {'compression_method': {'name': features}}
    """
    with torch.no_grad():
        db = {compression_method: {} for compression_method in compression_methods}
        pbar = tqdm.tqdm(os.listdir(db_path), desc="Building compressed face database")
        for name in pbar:
            if name.startswith('.') or name.startswith('_'):
                continue
            personal_path = os.path.join(db_path, name)
            for method in compression_methods:
                cfgs_str = ''
                for k, v in compression_cfgs[method].items():
                    cfgs_str += f"_{k}_{v}"
                f_name = os.path.join(personal_path, f"{fr.model_name}_{method}{cfgs_str}.pt")
                if not os.path.exists(f_name):
                    img_tensors = load_imgs(personal_path, img_sz=224, usage_portion=1.0, drange=1)
                    tmp_dl = DataLoader(img_tensors, batch_size=32, shuffle=False)
                    compressed_features = []
                    for imgs in tqdm.tqdm(tmp_dl, desc=f"Compressing and extracting features for {name} with {method} under the setting of {compression_cfgs[method]}"):
                        imgs = imgs.to(device)
                        compressed_features.append(fr(compress(imgs = imgs, method = method, differentiable = False, **compression_cfgs[method])).cpu())
                    compressed_features = torch.cat(compressed_features, dim=0)
                    db[method][name] = compressed_features.to(device)
                    with filelock.FileLock(f_name + ".lock"):
                        torch.save(compressed_features, f_name)
                else:
                    db[method][name] = torch.load(f_name, map_location=device, weights_only=False)
                    db[method][name] = db[method][name].to(device)
    return db
        
def cal_norms(x: torch.Tensor, y: torch.Tensor, epsilon: float) -> Dict[str, float]:
        """
        Calculate the norms between two tensors. Include l0, l1, l2, linf, and PSNR.

        Args:
            x (torch.Tensor): The first batch of images. Shape: [B, 3, H, W], range [0, 1], dtype torch.float32
            y (torch.Tensor): The second batch of images. Shape: [B, 3, H, W], range [0, 1], dtype torch.float32
            epsilon (float): The maximum perturbation value. 

        Returns:
            Dict[str, float]: A dictionary containing the norms.
        """
        resolution = x.size(2) * x.size(3)
        img_num = x.size(0)
        diffs = torch.abs(x - y)
        l0 = torch.sum(torch.any(diffs != 0, dim=1)).item() / (resolution * img_num)
        l1 = torch.sum(diffs).item() / (3*resolution*epsilon*img_num)
        l2 = torch.sum(torch.sqrt(diffs ** 2)).item() / img_num
        linf = torch.max(diffs).item()

        mse = torch.mean(diffs ** 2).item()
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)).item() if mse > 0 else float('inf')
        return {'l0': l0, 'l1': l1, 'l2': l2, 'linf': linf, 'psnr': psnr}

def retrieve(db: torch.Tensor, db_labels: List[str], queries: torch.Tensor, query_labels: List[str], dist_func: str, topk: int) -> List[float]:
    """
    Assume k queries in total. For each query, determine how many of the top-k retrievals are correct. 

    Args:
        db (torch.Tensor): The database of features. Shape: (N, D), where N is the number of features and D is the feature dimension.
        db_labels (List[str]): The labels of the database features.
        queries (torch.Tensor): The query features. Shape: (K, D), where K is the number of queries.
        query_labels (List[str]): The labels of the query features.
        dist_func (str): The distance function to use. Either 'cosine' or 'euclidean'.
        topk (int): The number of top retrievals to consider.

    Returns:
        List[float]: For each query, how many of the top-k retrievals are correct.
    """
    #k = queries.shape[0] # The number of queries
    k = topk
    if dist_func == 'cosine':
        matrix = torch.matmul(F.normalize(queries, p=2, dim=1), F.normalize(db, p=2, dim=1).T)
        db_matches_idxs = matrix.topk(k, dim=1, largest=True, sorted=False)[1]
    elif dist_func == 'euclidean':
        matrix = torch.cdist(queries, db, p=2)
        db_matches_idxs = matrix.topk(k, dim=1, largest=False, sorted=False)[1]
    else:
        raise ValueError(f"Unsupported distance function: {dist_func}. Use 'cosine' or 'euclidean'.")
    accus = []
    for i in range(queries.shape[0]):
        query_label = query_labels[i]
        db_matches = db_matches_idxs[i]
        correct_count = sum(1 for idx in db_matches if db_labels[idx] == query_label)
        accus.append(correct_count / k)
    return accus

def prot_eval(orig_features: torch.Tensor, protected_features: torch.Tensor, face_db: Dict[str, torch.Tensor], dist_func: str, query_portion: float, device: torch.device, verbose: bool = False) -> Dict[str, float]:
    """
    Protection evaluation function. It evaluates the retrieval accuracy of protected and unprotected queries against a database of facial features.

    Args:
        orig_features (torch.Tensor): The original features of the images. Shape: (N, D), where N is the number of features and D is the feature dimension.
        protected_features (torch.Tensor): The protected features of the images. Shape: (N, D).
        face_db (Dict[str, torch.Tensor]): The database of facial features.
        dist_func (str): The distance function to use. Either 'cosine' or 'euclidean'.
        query_portion (float): The portion of the images to use as queries.
        device (torch.device): The device to use for computation.

    Returns:
        Dict[str, float]: A dictionary containing the retrieval accuracies for different cases. 
    """
    noise_features, noise_labels = [], []
    for name, features in face_db.items():
        noise_features.append(features)
        noise_labels.extend([name] * features.shape[0])
    noise_features = torch.cat(noise_features, dim=0)

    img_num = orig_features.shape[0]
    query_num = int(img_num * query_portion)
    query_labels = ['_user'] * query_num
    db_labels = noise_labels + ['_user'] * (img_num - query_num)
    noise_features, orig_features, protected_features = noise_features.to(device), orig_features.to(device), protected_features.to(device)

    dist_func = dist_func
    #dist_func = fr.dis_func
    if verbose:
        print('===== Search with unprotected queries =====')    
    query_features = orig_features[:query_num]
    # (a) Protected entries should not be retrieved as top matches
    if verbose:
        print("Protected entries should not be retrieved as top matches")
    db_features = torch.cat([noise_features, protected_features[query_num:]], dim=0)
    one_as = retrieve(db=db_features, 
                     db_labels=db_labels, 
                     queries=query_features, 
                     query_labels=query_labels, 
                     dist_func=dist_func, 
                     topk=img_num - query_num)
    one_a = sum(one_as) / len(one_as)
    if verbose:
        print(f"1A Retrieval accuracy: {one_a:.4f} (lower is better)")
    # (b) Unprotected entries are deemed to be retrieved as top matches
    if verbose:
        print("Unprotected entries are deemed to be retrieved as top matches")
    db_features = torch.cat([noise_features, orig_features[query_num:]], dim=0)
    one_bs = retrieve(db=db_features, 
                     db_labels=db_labels, 
                     queries=query_features, 
                     query_labels=query_labels, 
                     dist_func=dist_func, 
                     topk=img_num - query_num)
    one_b = sum(one_bs) / len(one_bs)
    if verbose:
        print(f"1B Retrieval accuracy: {one_b:.4f} (higher is better)")

    # Case 2: Search with protected queries
    if verbose:
        print('===== Search with protected queries =====')
    query_features = protected_features[:query_num]
    # (a) Protected entries should not be retrieved as top matches
    if verbose:
        print("Protected entries should not be retrieved as top matches")
    db_features = torch.cat([noise_features, protected_features[query_num:]], dim=0)
    two_as = retrieve(db=db_features, 
                     db_labels=db_labels, 
                     queries=query_features, 
                     query_labels=query_labels, 
                     dist_func=dist_func, 
                     topk=img_num - query_num)
    two_a = sum(two_as) / len(two_as)
    if verbose:
        print(f"2A Retrieval accuracy: {two_a:.4f} (lower is better)")
    # (b) Unprotected entries should not be retrieved as top matches
    if verbose:
        print("Unprotected entries should not be retrieved as top matches")
    db_features = torch.cat([noise_features, orig_features[query_num:]], dim=0)
    two_bs = retrieve(db=db_features, 
                     db_labels=db_labels, 
                     queries=query_features, 
                     query_labels=query_labels, 
                     dist_func=dist_func, 
                     topk=img_num - query_num)
    two_b = sum(two_bs) / len(two_bs)
    if verbose:
        print(f"2B Retrieval accuracy: {two_b:.4f} (lower is better)")

    return {'1a': one_a,'1b': one_b,'2a': two_a,'2b': two_b}

def eval_masks(three_d: bool, 
            test_data: DataLoader, 
            face_db: Dict[str, torch.Tensor], 
            fr: FR, 
            device: torch.device, 
            bin_mask: bool, 
            epsilon: float, 
            masks: torch.Tensor,
            query_portion: float = 0.5, 
            vis_eval: bool = True, 
            verbose: bool = False) -> Dict[str, float]:
    """
    Evaluate masks against a database of facial features.

    Args:
        three_d (bool): Whether to use 3D masks.
        test_data (DataLoader): The DataLoader containing the test data.
        face_db (Dict[str, torch.Tensor]): The database of facial features.
        fr (FR): The facial recognition model.
        device (torch.device): The device to use for computation.
        bin_mask (bool): Whether to use binary masks.
        epsilon (float): The maximum perturbation value.
        masks (torch.Tensor): The masks to apply. Shape: (B, 3, H, W). Range: [-epsilon, epsilon]. dtype: np.float32.
        query_portion (float): The portion of the test data to use as queries.
        vis_eval (bool): Whether to visualize the evaluation results.
        verbose (bool): Whether to print the evaluation results.

    Returns:
        Dict[str, float]: A dictionary containing the evaluation results. Includes retrieval accuracies, various norms, and visual quality metrics.
    """
    tensor_masks = masks
    img_cnt = 0
    orig_features, protected_features = [], []
    ssims, psnrs, l0s, l1s, l2s, linfs = [], [], [], [], [], []
    for idx, tensors in tqdm.tqdm(enumerate(test_data), total=len(test_data), desc="Applying mask, extracting features, and evaluating visual quality"):
        orig_faces = tensors[0].to(device)  # Shape: [B, 3, H, W]
        img_num = orig_faces.shape[0]
        if three_d:
            uvs = tensors[1].to(device)
            textures = tensor_masks[img_cnt:img_cnt + img_num].to(device)  # Shape: [B, 224, 224, 3]
            perturbations = torch.clamp(F.grid_sample(textures, uvs, mode='bilinear', align_corners=True), -epsilon, epsilon)
        else:
            perturbations = tensor_masks[img_cnt:img_cnt + img_num].to(device)
        if bin_mask:
            bin_masks = tensors[2].to(device)
            perturbations *= bin_masks
        protected_faces = torch.clamp(orig_faces + perturbations, 0, 1)
        img_cnt += img_num
        orig_features.append(fr(orig_faces).cpu())
        protected_features.append(fr(protected_faces).cpu())
        for prot_img, orig_img in zip(protected_faces, orig_faces):
            prot_img = prot_img.unsqueeze(0)
            orig_img = orig_img.unsqueeze(0)
            if vis_eval:
                ssims.append(calc_ssim(prot_img, orig_img, data_range=1, size_average=True).cpu().numpy().item())
                norms = cal_norms(prot_img, orig_img, epsilon)
                l0s.append(norms['l0'])
                l1s.append(norms['l1'])
                l2s.append(norms['l2'])
                linfs.append(norms['linf'])
                psnrs.append(norms['psnr'])
    del orig_faces, protected_faces, perturbations
    complete_del()

    if not vis_eval:
        psnrs, ssims, l0s, l1s, l2s, linfs = [0], [0], [0], [0], [0], [0]
    mean_l0 = np.mean(l0s).item()
    max_l0 = max(l0s)
    min_l0 = min(l0s)
    mean_l1 = np.mean(l1s).item()
    max_l1 = max(l1s)
    min_l1 = min(l1s)
    mean_l2 = np.mean(l2s).item()
    max_l2 = max(l2s)
    min_l2 = min(l2s)
    mean_linf = np.mean(linfs).item() * 255
    max_linf = max(linfs) * 255
    min_linf = min(linfs) * 255
    mean_ssim = np.mean(ssims).item()
    max_ssim = max(ssims)
    min_ssim = min(ssims)
    mean_psnr = np.mean(psnrs).item()
    max_psnr = max(psnrs)
    min_psnr = min(psnrs)
    orig_features = torch.cat(orig_features, dim=0).to(device)
    protected_features = torch.cat(protected_features, dim=0).to(device)

    prot_res = prot_eval(orig_features=orig_features,
                         protected_features=protected_features,
                         face_db=face_db,
                         dist_func='cosine',
                         query_portion=query_portion,
                         device=device,
                         verbose=verbose)
    one_a = prot_res['1a']
    one_b = prot_res['1b']
    two_a = prot_res['2a']
    two_b = prot_res['2b']

    return {
        '1a': one_a,
        '1b': one_b,
        '2a': two_a,
        '2b': two_b,
        'l0': mean_l0,
        'l0_max': max_l0,
        'l0_min': min_l0,
        'l1': mean_l1,
        'l1_max': max_l1,
        'l1_min': min_l1,
        'l2': mean_l2,
        'l2_max': max_l2,
        'l2_min': min_l2,
        'linf': mean_linf,
        'linf_max': max_linf,
        'linf_min': min_linf,
        'ssim': mean_ssim,
        'ssim_max': max_ssim,
        'ssim_min': min_ssim,
        'psnr': mean_psnr,
        'psnr_max': max_psnr,
        'psnr_min': min_psnr
    }

def compression_eval(compression_methods: List[str], 
                    compression_cfgs: Dict[str, Dict[str, Any]],
                    three_d: bool, 
                    test_data: DataLoader, 
                    face_db: Dict[str, Dict[str, torch.Tensor]], 
                    fr: FR, 
                    device: torch.device, 
                    bin_mask: bool, 
                    epsilon: float, 
                    masks: np.ndarray, 
                    query_portion: float = 0.5) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the masks' robustness against different compression methods.

    Args:
        compression_methods (List[str]): The list of compression methods to evaluate.
        compression_cfgs (Dict[str, Dict[str, Any]]): The configurations for each compression method.
        three_d (bool): Whether to use 3D masks.
        test_data (DataLoader): The DataLoader containing the test data.
        face_db (Dict[str, Dict[str, torch.Tensor]]): The database of facial features for different compression methods.
        fr (FR): The facial recognition model.
        device (torch.device): The device to use for computation.
        bin_mask (bool): Whether to use binary masks.
        epsilon (float): The maximum perturbation value.
        masks (np.ndarray): The masks to apply. Shape: (B, 3, H, W). Range: [-epsilon, epsilon]. dtype: np.float32.
        query_portion (float): The portion of the test data to use as queries.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary containing the evaluation results for each compression method.
        Each method's results include '1a', '1b', '2a', '2b'.
    """
    
    tensor_masks = torch.tensor(masks, dtype=torch.float32, device=device)
    prot_res = {}
    for method in compression_methods:
        img_cnt = 0
        orig_features, protected_features = [], []
        for idx, tensors in tqdm.tqdm(enumerate(test_data), total=len(test_data), desc=f"Compressing imgs with {method}, Applying mask, and extracting features"):
            orig_faces = tensors[0].to(device)  # Shape: [B, 3, H, W]
            img_num = orig_faces.shape[0]
            if three_d:
                uvs = tensors[1].to(device)
                textures = tensor_masks[img_cnt:img_cnt + img_num].to(device)  # Shape: [B, 224, 224, 3]
                perturbations = torch.clamp(F.grid_sample(textures, uvs, mode='bilinear', align_corners=True), -epsilon, epsilon)
            else:
                perturbations = tensor_masks[img_cnt:img_cnt + img_num].to(device)
            if bin_mask:
                bin_masks = tensors[2].to(device)
                perturbations *= bin_masks
            protected_faces = torch.clamp(orig_faces + perturbations, 0, 1)
            compressed_protected_faces = compress(imgs = protected_faces, method = method, differentiable = False, **compression_cfgs[method])
            compressed_orig_faces = compress(imgs = orig_faces, method = method, differentiable = False, **compression_cfgs[method])

            img_cnt += img_num
            orig_features.append(fr(compressed_orig_faces).cpu())
            protected_features.append(fr(compressed_protected_faces).cpu())
            for prot_img, orig_img in zip(protected_faces, orig_faces):
                prot_img = prot_img.unsqueeze(0)
                orig_img = orig_img.unsqueeze(0)
        del orig_faces, protected_faces, perturbations
        complete_del()
        orig_features = torch.cat(orig_features, dim=0).to(device)
        protected_features = torch.cat(protected_features, dim=0).to(device)

        prot_res[method] = prot_eval(orig_features=orig_features,
                                        protected_features=protected_features,
                                        face_db=face_db[method],
                                        dist_func='cosine',
                                        query_portion=query_portion,
                                        device=device)
    return prot_res

def visualize_mask(
    orig_img: torch.Tensor,
    uv: torch.Tensor,
    bin_mask: torch.Tensor,
    univ_mask: torch.Tensor,
    save_path: str,
    epsilon: float, 
    use_bin_mask: bool,
    three_d: bool
) -> None:
    _orig_img = orig_img.detach()
    _uv = uv.detach()
    _bin_mask = bin_mask.detach()
    _univ_mask = univ_mask.detach()
    if three_d:
        _pert = torch.clamp(F.grid_sample(_univ_mask, _uv.unsqueeze(0), mode='bilinear', align_corners=True).squeeze(0), -epsilon, epsilon)
    else:
        _pert = _univ_mask.squeeze(0)
    if use_bin_mask:
        _pert = _pert * _bin_mask
    _prot_img = torch.clamp(_orig_img + _pert, 0, 1)

    _orig_img = _orig_img.permute(1, 2, 0).mul(255.).cpu().contiguous().numpy().astype(np.uint8)
    _prot_img = _prot_img.permute(1, 2, 0).mul(255.).cpu().contiguous().numpy().astype(np.uint8)
    _pert = (_pert - _pert.min()) / (_pert.max() - _pert.min())
    if use_bin_mask:
        _pert = _pert * _bin_mask
    _pert = _pert.detach().permute(1, 2, 0).mul(255.).cpu().contiguous().numpy().astype(np.uint8)
    _univ_mask = (_univ_mask - _univ_mask.min()) / (_univ_mask.max() - _univ_mask.min())
    _univ_mask = _univ_mask.squeeze(0).permute(1, 2, 0).mul(255.).cpu().contiguous().numpy().astype(np.uint8)

    _uv = (_uv - _uv.min()) / (_uv.max() - _uv.min())
    _uv = _uv.mul(255).cpu().contiguous().numpy().astype(np.uint8)
    _uv_img = np.zeros((_uv.shape[0], _uv.shape[1], 3), dtype=np.uint8)
    _uv_img[..., 1:] = _uv
    _uv = _uv_img
    # Turn binary mask into a black-white image
    _bin_mask = _bin_mask.mul(255).permute(1, 2, 0).cpu().contiguous().numpy().astype(np.uint8)
    _bin_mask = cv2.cvtColor(_bin_mask, cv2.COLOR_GRAY2RGB)

    plt.figure(figsize=(12, 8))
    for i, (_img, _title) in enumerate(zip([_orig_img, _prot_img, _univ_mask, _pert, _uv, _bin_mask], 
                                        ['Original Image', 'Protected Image', 'Universal Mask', 'Perturbation', f'UV({"Used" if three_d else "Unused"})', f'Binary Mask({"Used" if use_bin_mask else "Unused"})'])):
        plt.subplot(2, 3, i+1)
        plt.imshow(_img)
        plt.title(_title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
