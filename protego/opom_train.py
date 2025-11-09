import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" 
from typing import List

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_msssim import ssim as calc_ssim
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import cvxpy as cp

from .FacialRecognition import FR

class ConvexHullLoss(nn.Module):
    def __init__(self):
        super(ConvexHullLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, prot_feats: torch.Tensor, orig_feats: torch.Tensor, upper: float, lower: float) -> torch.Tensor:
        normalized_prot = F.normalize(prot_feats, p=2, dim=1)
        normalized_orig = F.normalize(orig_feats, p=2, dim=1)
        orig_feats_np = normalized_orig.clone().detach().cpu().numpy()
        coeffs_lst = []
        for img_idx in range(normalized_prot.shape[0]):
            prot_feat_np = normalized_prot[img_idx].clone().detach().cpu().numpy()
            coeff = cp.Variable(orig_feats_np.shape[0])
            objective = cp.Minimize(cp.sum_squares(coeff @ orig_feats_np - prot_feat_np))
            constraints = [sum(coeff) == 1, coeff >= lower, coeff <= upper]
            problem = cp.Problem(objective, constraints)
            problem.solve()
            coeffs_lst.append(torch.tensor(coeff.value, dtype=torch.float32, device=prot_feats.device))
        coeffs = torch.stack(coeffs_lst, dim=0)
        dis = self.mse(torch.mm(coeffs, normalized_orig), normalized_prot)
        return dis

def train_opom_mask(cfgs: OmegaConf, 
                    frs: List[FR], 
                    train_dl: DataLoader, 
                    results_save_path: str = None) -> torch.Tensor:
    """
    Train the protection mask using OPOM method.

    Args:
        cfgs (OmegaConf): The configuration object containing training parameters.
        frs (List[FR]): List of facial recognition models to use for feature extraction.
        train_dl (DataLoader): The dataloader for training.
        results_save_path (str): The path to save the results. If None, the training loss plots will not be saved.

    Returns:
        torch.Tensor: The trained universal mask of shape [1, 3, mask_size, mask_size].
    """
    epoch_num = cfgs.epoch_num
    learning_rate = cfgs.learning_rate * 255.
    epsilon = cfgs.epsilon * 255.
    mask_size = cfgs.mask_size
    device = frs[0].device
    mask_random_seed = cfgs.mask_random_seed

    # Init the mask
    mask_rand_generator = torch.Generator(device=device)
    mask_rand_generator.manual_seed(mask_random_seed)
    univ_mask = (torch.rand(size=(1, 3, mask_size, mask_size), device=device, requires_grad=False, generator=mask_rand_generator, dtype=torch.float32) - 0.5) * epsilon / 0.5

    # Init the recorders
    losses = []

    # Init hyperparameters
    init_iter = 10
    hull_upper_bound = 1.0
    hull_lower_bound = 0.0

    loss_fn = ConvexHullLoss()
    loss_fn.to(device)

    for epoch in range(epoch_num):
        pbar = tqdm.tqdm(enumerate(train_dl), total=len(train_dl))
        batch_losses = []
        for batch_idx, tensors in pbar:
            orig_faces = tensors[0].to(device).mul(255.)
            uvs = tensors[1].to(device)
            if cfgs.bin_mask:
                bin_masks = tensors[2].to(device)
            img_num = orig_faces.shape[0]
            perturbations = univ_mask.repeat(img_num, 1, 1, 1)
            if cfgs.bin_mask:
                restricted_perts = perturbations * bin_masks
                protected_faces = torch.clamp(orig_faces + restricted_perts, 0, 255.)
            else:
                protected_faces = torch.clamp(orig_faces + perturbations, 0, 255.)
            protected_faces.requires_grad_(True)
            ensemble_losses, cos_sims = [], []
            for fr in frs:
                protected_features = fr(protected_faces)
                orig_features = fr(orig_faces)
                cos_sims.append(torch.mean(F.cosine_similarity(protected_features, orig_features, dim=1)))
                if epoch < init_iter:
                    loss = loss_fn(protected_features, orig_features, upper=1/img_num, lower=1/img_num)
                else:
                    loss = loss_fn(protected_features, orig_features, upper=hull_upper_bound, lower=hull_lower_bound)
                ensemble_losses.append(loss)
            overall_loss = torch.mean(torch.stack(ensemble_losses, dim=0))
            cos_sim = torch.mean(torch.stack(cos_sims, dim=0), dim=0)
            overall_loss.backward()

            grad_avg = torch.mean(protected_faces.grad.detach(), dim=0, keepdim=True)
            protected_faces = protected_faces.detach() + learning_rate * torch.sign(grad_avg)
            univ_mask = torch.clamp(torch.mean(protected_faces - orig_faces, dim=0, keepdim=True), -epsilon, epsilon)

            overall_loss = overall_loss.detach().cpu().numpy().item()
            cos_sim = cos_sim.detach().cpu().numpy().item()
            batch_losses.append(overall_loss)
            pbar.set_description(f"Epoch {epoch+1}/{epoch_num} Loss: {overall_loss:.4f} Cosine Similarity: {cos_sim:.4f}")
        losses.append(np.mean(batch_losses))

    if results_save_path is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, epoch_num+1), losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('OPOM Training Loss Curve')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(results_save_path, 'opom_training_loss_curve.png'))
        plt.close()

    return univ_mask.detach().div(255.).cpu()
