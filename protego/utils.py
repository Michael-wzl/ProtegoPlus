import os
from typing import Dict, List, Tuple, Any
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

from .FacialRecognition import FR
from .compression import compress

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def complete_del() -> None:
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.backends.mps.empty_cache() if torch.backends.mps.is_available() else None

def img2tensor(img: np.ndarray, drange: int = 1) -> torch.Tensor:
    """
    Convert a numpy image to a tensor.

    Args:
        img (np.ndarray): The numpy image to convert. Shape: (H, W, 3), dtype: uint8, range [0, 255].
        drange (int): The range of the image. Default is 1. Either 1 or 255.

    Returns:
        torch.Tensor: The converted tensor. Shape: (1, 3, H, W), dtype: float32., range [0, 1].
    """
    img_tensor = torch.tensor(img, dtype=torch.float32, device=torch.device('cpu'))
    img_tensor = img_tensor.permute(2, 0, 1)
    if drange == 1:
        img_tensor /= 255. 
    img_tensor = img_tensor.unsqueeze(0)
    return torch.clamp(img_tensor, 0, 1)

def load_imgs(base_dir: str, img_sz: int, usage_portion: float = 1.0, drange: int = 1) -> torch.Tensor:
    """
    Read images from a directory and convert them to tensors.

    Args:
        base_dir (str): The directory containing the images.
        img_sz (int): The size of the image.
        usage_portion (float): The portion of the images to use.
        drange (int): The range of the image. Default is 1. Either 1 or 255.

    Returns:
        torch.Tensor: The tensor of the images. Shape [B, 3, img_sz, img_sz]. RGB, range [0, 1], dtype: float32.
    """
    img_tensors = []
    img_num = len(os.listdir(base_dir))
    imgs_names = [name for name in os.listdir(base_dir) if name.endswith(('.jpg', '.png', '.jpeg', '.bmp')) and not (name.startswith('.') or name.startswith('_'))]
    for file_name in sorted(imgs_names):
        img_path = os.path.join(base_dir, file_name)
        img = PIL.Image.open(img_path).convert('RGB')
        img = np.array(img)
        img = cv2.resize(img, (img_sz, img_sz), interpolation=cv2.INTER_LANCZOS4 if img.shape[0] < img_sz else cv2.INTER_AREA)
        img_tensors.append(img2tensor(img, drange=drange))
        if len(img_tensors) >= int(img_num * usage_portion):
            break
    return torch.cat(img_tensors, dim=0)

def load_uvs(base_dir: str, usage_portion: float = 1.0) -> torch.Tensor:
    """
    Read UV maps from a directory. 

    Args:
        base_dir (str): The directory containing the UV maps.
        usage_portion (float): The portion of the UV maps to use.

    Returns:
        torch.Tensor: The tensor of the UV maps. Shape [B, 224, 224, 2]. Range [-1, 1], dtype: float32.

    Raises:
        FileNotFoundError: If the UV maps file does not exist in the specified directory.
    """
    uvs_path = os.path.join(base_dir, 'uvs.pt')
    if not os.path.exists(uvs_path):
        raise FileNotFoundError(f"UV maps not found in {uvs_path}. Please generate UV maps first.")
    uvs = torch.load(uvs_path, weights_only=False)    
    img_num = uvs.shape[0]
    if usage_portion < 1.0:
        uvs = uvs[:int(img_num * usage_portion)]
    return uvs

def load_bin_masks(base_dir: str, usage_portion: float = 1.0) -> torch.Tensor:
    """
    Read binary masks from a directory.

    Args:
        base_dir (str): The directory containing the binary masks.
        usage_portion (float): The portion of the UV maps to use.

    Returns:
        torch.Tensor: The tensor of the binary masks. Shape [B, 1, H, W]. Range [0, 1], dtype: float32.

    Raises:
        FileNotFoundError: If the binary masks file does not exist in the specified directory.
    """
    bin_mask_path = os.path.join(base_dir, 'visibility_masks.pt')
    if not os.path.exists(bin_mask_path):
        raise FileNotFoundError(f"Binary Masks not found in {bin_mask_path}. Please generate Binary Masks first.")
    bin_masks = torch.load(bin_mask_path, weights_only=False)
    img_num = bin_masks.shape[0]
    if usage_portion < 1.0:
        bin_masks = bin_masks[:int(img_num * usage_portion)]
    return bin_masks.permute(0, 3, 1, 2).float()  # Convert to [B, 1, H, W] format

def split(imgs: torch.Tensor, uvs: torch.Tensor, bin_masks: torch.Tensor, train_portion: float, random_split: bool, shuffle: bool, batch_size: int, num_threads: int = 4, pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Split the dataset into a training and validation set.

    Args:
        imgs (torch.Tensor): The tensor of the images. Shape [B, 3, H, W].
        uvs (torch.Tensor): The tensor of the UV maps. Shape [B, 224, 224, 2]. You may set it to None. 
        bin_masks (torch.Tensor): The tensor of the binary masks. Shape [B, 224, 224, 1]. You may set it to None. 
        train_portion (float): The portion of the each person's images to be used as the training set.
        random_split (bool): Whether to randomly split the dataset. If True, the images will be shuffled before splitting.
        batch_size (int): The batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the datasets.
        num_threads (int): The number of threads to use for the DataLoader. Default is 4.
        pin_memory (bool): Whether to pin memory for the DataLoader. Default is True.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the DataLoader for the training and validation set. The validation set's batch size is set to 32.

    Raises:
        ValueError: If the number of images, UV maps, and binary masks do not match. Or if the specified data combinations are not allowed.
    """
    
    if bin_masks is None and uvs is None:
        dataset = TensorDataset(imgs)
    elif bin_masks is None and uvs is not None:
        assert imgs.shape[0] == uvs.shape[0], "The number of images and UV maps must be the same."
        dataset = TensorDataset(imgs, uvs)
    elif bin_masks is not None and uvs is None:
        assert imgs.shape[0] == bin_masks.shape[0], "The number of images and binary masks must be the same."
        dataset = TensorDataset(imgs, bin_masks)
    elif bin_masks is not None and uvs is not None:
        assert imgs.shape[0] == uvs.shape[0] == bin_masks.shape[0], "The number of images, UV maps, and binary masks must be the same."
        dataset = TensorDataset(imgs, uvs, bin_masks)
    else:
        raise ValueError("Allowed combinations of inputs: (imgs), (imgs, bin_masks), (imgs, uvs), (imgs, uvs, bin_masks).")
    total_size = len(dataset)
    train_size = int(total_size * train_portion)
    val_size = total_size - train_size

    if random_split:
        train_dataset, val_dataset = train_val_split(dataset, [train_size, val_size])
    else:
        if bin_masks is None and uvs is None:
            train_dataset = TensorDataset(imgs[:train_size])
            val_dataset = TensorDataset(imgs[train_size:])
        elif bin_masks is None and uvs is not None:
            train_dataset = TensorDataset(imgs[:train_size], uvs[:train_size])
            val_dataset = TensorDataset(imgs[train_size:], uvs[train_size:])
        elif bin_masks is not None and uvs is None:
            train_dataset = TensorDataset(imgs[:train_size], bin_masks[:train_size])
            val_dataset = TensorDataset(imgs[train_size:], bin_masks[train_size:])
        elif bin_masks is not None and uvs is not None:
            train_dataset = TensorDataset(imgs[:train_size], uvs[:train_size], bin_masks[:train_size])
            val_dataset = TensorDataset(imgs[train_size:], uvs[train_size:], bin_masks[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_threads, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=shuffle, num_workers=num_threads, pin_memory=pin_memory)

    return train_loader, val_loader

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
            x (torch.Tensor): The first batch of images.
            y (torch.Tensor): The second batch of images.
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

def retrieve(db: torch.Tensor, db_labels: List[str], queries: torch.Tensor, query_labels: List[str], dist_func: str) -> List[float]:
    """
    Assume k queries in total. For each query, determine how many of the top-k retrievals are correct. 

    Args:
        db (torch.Tensor): The database of features. Shape: (N, D), where N is the number of features and D is the feature dimension.
        db_labels (List[str]): The labels of the database features.
        queries (torch.Tensor): The query features. Shape: (K, D), where K is the number of queries.
        query_labels (List[str]): The labels of the query features.
        dist_func (str): The distance function to use. Either 'cosine' or 'euclidean'.

    Returns:
        List[float]: For each query, how many of the top-k retrievals are correct.
    """
    k = queries.shape[0] # The number of queries
    if dist_func == 'cosine':
        matrix = torch.matmul(F.normalize(queries, p=2, dim=1), F.normalize(db, p=2, dim=1).T)
        db_matches_idxs = matrix.topk(k, dim=1, largest=True, sorted=False)[1]
    elif dist_func == 'euclidean':
        matrix = torch.cdist(queries, db, p=2)
        db_matches_idxs = matrix.topk(k, dim=1, largest=False, sorted=False)[1]
    else:
        raise ValueError(f"Unsupported distance function: {dist_func}. Use 'cosine' or 'euclidean'.")
    accus = []
    for i in range(k):
        query_label = query_labels[i]
        db_matches = db_matches_idxs[i]
        correct_count = sum(1 for idx in db_matches if db_labels[idx] == query_label)
        accus.append(correct_count / k)
    return accus

def prot_eval(orig_features: torch.Tensor, protected_features: torch.Tensor, face_db: Dict[str, torch.Tensor], dist_func: str, query_portion: float, device: torch.device) -> Dict[str, float]:
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
    # print('===== Search with unprotected queries =====')    
    query_features = orig_features[:query_num]
    # (a) Protected entries should not be retrieved as top matches
    # print("Protected entries should not be retrieved as top matches")
    db_features = torch.cat([noise_features, protected_features[query_num:]], dim=0)
    one_as = retrieve(db=db_features, 
                     db_labels=db_labels, 
                     queries=query_features, 
                     query_labels=query_labels, 
                     dist_func=dist_func)
    one_a = sum(one_as) / len(one_as)
    # print(f"1A Retrieval accuracy: {one_a:.4f} (lower is better)")
    # (b) Unprotected entries are deemed to be retrieved as top matches
    # print("Unprotected entries are deemed to be retrieved as top matches")
    db_features = torch.cat([noise_features, orig_features[query_num:]], dim=0)
    one_bs = retrieve(db=db_features, 
                     db_labels=db_labels, 
                     queries=query_features, 
                     query_labels=query_labels, 
                     dist_func=dist_func)
    one_b = sum(one_bs) / len(one_bs)
    # print(f"1B Retrieval accuracy: {one_b:.4f} (higher is better)")

    # Case 2: Search with protected queries
    # print('===== Search with protected queries =====')
    query_features = protected_features[:query_num]
    # (a) Protected entries should not be retrieved as top matches
    # print("Protected entries should not be retrieved as top matches")
    db_features = torch.cat([noise_features, protected_features[query_num:]], dim=0)
    two_as = retrieve(db=db_features, 
                     db_labels=db_labels, 
                     queries=query_features, 
                     query_labels=query_labels, 
                     dist_func=dist_func)
    two_a = sum(two_as) / len(two_as)
    # print(f"2A Retrieval accuracy: {two_a:.4f} (lower is better)")
    # (b) Unprotected entries should not be retrieved as top matches
    # print("Unprotected entries should not be retrieved as top matches")
    db_features = torch.cat([noise_features, orig_features[query_num:]], dim=0)
    two_bs = retrieve(db=db_features, 
                     db_labels=db_labels, 
                     queries=query_features, 
                     query_labels=query_labels, 
                     dist_func=dist_func)
    two_b = sum(two_bs) / len(two_bs)
    # print(f"2B Retrieval accuracy: {two_b:.4f} (lower is better)")

    return {'1a': one_a,'1b': one_b,'2a': two_a,'2b': two_b}

def eval_masks(three_d: bool, 
            test_data: DataLoader, 
            face_db: Dict[str, torch.Tensor], 
            fr: FR, 
            device: torch.device, 
            bin_mask: bool, 
            epsilon: float, 
            masks: np.ndarray, 
            query_portion: float = 0.5, 
            vis_eval: bool = True) -> Dict[str, float]:
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
        masks (np.ndarray): The masks to apply. Shape: (B, 3, H, W). Range: [-epsilon, epsilon]. dtype: np.float32.
        query_portion (float): The portion of the test data to use as queries.
        vis_eval (bool): Whether to visualize the evaluation results.

    Returns:
        Dict[str, float]: A dictionary containing the evaluation results. Includes retrieval accuracies, various norms, and visual quality metrics.
    """
    tensor_masks = torch.tensor(masks, dtype=torch.float32, device=device)
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
                         device=device)
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

def visualize_3dmask(orig_img: torch.Tensor, 
                    uv: torch.Tensor,
                    save_path: str, 
                    epsilon: float,
                    univ_texture: torch.Tensor = None, 
                    additional_texture: torch.Tensor = None, 
                    bin_mask: torch.Tensor = None) -> None: 
    """
    Visualize the 3D mask on the original image and save the result.

    Args:
        orig_img (torch.Tensor): The original image. Shape: (1, 3, H, W).
        uv (torch.Tensor): The UV map. Shape: (1, H, W, 2).
        save_path (str): The path to save the visualization.
        epsilon (float): The maximum perturbation value.
        univ_texture (torch.Tensor): The universal texture to visualize. Shape: (1, 3, H, W)
        additional_texture (torch.Tensor): Additional texture to visualize. Shape: (1, 3, H, W)
        bin_mask (torch.Tensor): The binary mask to apply. Shape: (1, 1, H, W). If None, no binary mask is applied.
    """
    assert not (univ_texture is None and additional_texture is None), "Either univ_texture or additional_texture must be provided."
    if univ_texture is not None and additional_texture is not None:
        texture = torch.clamp(univ_texture + additional_texture, -epsilon, epsilon)
    elif univ_texture is not None and additional_texture is None:
        texture = univ_texture
    elif univ_texture is None and additional_texture is not None:
        texture = additional_texture
    
    perturbation = torch.clamp(F.grid_sample(texture, uv, mode='bilinear', align_corners=True), -epsilon, epsilon)
    if bin_mask is not None:
        perturbation *= bin_mask
    protected_img = torch.clamp(orig_img + perturbation, 0, 1)

    ## print(texture.max()*255., texture.min()*255., perturbation.max()*255., perturbation.min()*255., protected_img.max()*255., protected_img.min()*255.)
    orig_img_np = (orig_img.squeeze(0).permute(1, 2, 0)*255.).cpu().numpy().astype(np.uint8)
    protected_img_np = (protected_img.squeeze(0).permute(1, 2, 0)*255.).cpu().numpy().astype(np.uint8)
    perturbation_np = (torch.clamp((perturbation - perturbation.min()) / (perturbation.max() - perturbation.min()), 0, 1)*255.).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    texture_np = (torch.clamp((texture - texture.min()) / (texture.max() - texture.min()), 0, 1)*255.).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    if univ_texture is not None:
        univ_texture_np = (torch.clamp((univ_texture - univ_texture.min()) / (univ_texture.max() - univ_texture.min()), 0, 1)*255.).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) 
    else:
        univ_texture_np = np.zeros((224, 224, 3), dtype=np.uint8)
    if additional_texture is not None:
        additional_texture_np = (torch.clamp((additional_texture - additional_texture.min()) / (additional_texture.max() - additional_texture.min()), 0, 1)*255.).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) 
    else:
        additional_texture_np = np.zeros((224, 224, 3), dtype=np.uint8)

    plt.subplot(2, 3, 1)
    plt.imshow(orig_img_np)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(2, 3, 2)
    plt.imshow(protected_img_np)
    plt.title('Protected Image')
    plt.axis('off')
    plt.subplot(2, 3, 3)
    plt.imshow(perturbation_np)
    plt.title('Perturbation')
    plt.axis('off')
    plt.subplot(2, 3, 4)
    plt.imshow(univ_texture_np)
    plt.title('Universal Texture')
    plt.axis('off')
    plt.subplot(2, 3, 5)
    plt.imshow(additional_texture_np)
    plt.title('Additional Texture')
    plt.axis('off')
    plt.subplot(2, 3, 6)
    plt.imshow(texture_np)
    plt.title('Final Texture')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_2dmask(orig_img: torch.Tensor, 
                    save_path: str, 
                    epsilon: float,
                    univ_mask: torch.Tensor = None, 
                    bin_mask: torch.Tensor = None, 
                    additional_mask: torch.Tensor = None) -> None: 
    """
    Visualize the 3D mask on the original image and save the result.

    Args:
        orig_img (torch.Tensor): The original image. Shape: (1, 3, H, W).
        save_path (str): The path to save the visualization.
        epsilon (float): The maximum perturbation value.
        univ_mask (torch.Tensor): The universal mask to visualize. Shape: (1, 3, H, W)
        additional_mask (torch.Tensor): Additional mask to visualize. Shape: (1, 3, H, W)
        bin_mask (torch.Tensor): The binary mask to apply. Shape: (1, 1, H, W). If None, no binary mask is applied.
    """
    assert not (univ_mask is None and additional_mask is None), "Either univ_mask or additional_mask must be provided."
    if univ_mask is not None and additional_mask is not None:
        perturbation = torch.clamp(univ_mask + additional_mask, -epsilon, epsilon)
    elif univ_mask is not None and additional_mask is None:
        perturbation = univ_mask
    elif univ_mask is None and additional_mask is not None:
        perturbation = additional_mask
    if bin_mask is not None:
        perturbation *= bin_mask
    protected_img = torch.clamp(orig_img + perturbation, 0, 1)

    ## print(perturbation.max()*255., perturbation.min()*255., protected_img.max()*255., protected_img.min()*255.)
    orig_img_np = (orig_img.squeeze(0).permute(1, 2, 0)*255.).cpu().numpy().astype(np.uint8)
    protected_img_np = (protected_img.squeeze(0).permute(1, 2, 0)*255.).cpu().numpy().astype(np.uint8)
    perturbation_np = (torch.clamp((perturbation - perturbation.min()) / (perturbation.max() - perturbation.min()), 0, 1)*255.).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    if univ_mask is not None:
        univ_mask_np = (torch.clamp((univ_mask - univ_mask.min()) / (univ_mask.max() - univ_mask.min()), 0, 1)*255.).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) 
    else:
        univ_mask_np = np.zeros((224, 224, 3), dtype=np.uint8)
    if additional_mask is not None:
        additional_mask_np = (torch.clamp((additional_mask - additional_mask.min()) / (additional_mask.max() - additional_mask.min()), 0, 1)*255.).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    else:
        additional_mask_np = np.zeros((224, 224, 3), dtype=np.uint8)
    
    plt.subplot(2, 3, 1)
    plt.imshow(orig_img_np)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(2, 3, 2)
    plt.imshow(protected_img_np)
    plt.title('Protected Image')
    plt.axis('off')
    plt.subplot(2, 3, 3)
    plt.imshow(perturbation_np)
    plt.title('Perturbation')
    plt.axis('off')
    plt.subplot(2, 3, 4)
    plt.imshow(univ_mask_np)
    plt.title('Universal Perturbation')
    plt.axis('off')
    plt.subplot(2, 3, 5)
    plt.imshow(additional_mask_np)
    plt.title('Additional Perturbation')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
