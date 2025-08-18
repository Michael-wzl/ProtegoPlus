from typing import Dict, Any

import torch
import torch.nn.functional as F

from DiffJPEG.DiffJPEG import DiffJPEG

def gaussian_filter(imgs:torch.Tensor, kernel_size:int=5, sigma:float=1.0, differentiable: bool = True) -> torch.Tensor:
    """
    A differentiable Gaussian filter

    Args:
        imgs (Tensor): Input image tensor of shape [B, C, H, W], Range: [0, 1]
        kernel_size (int): Size of the Gaussian kernel (must be odd)
        sigma (float): Standard deviation for Gaussian distribution
        differentiable (bool): Placeholder for differentiable flag, not used in this implementation

    Returns:
        Tensor: Filtered image tensor of shape [B, C, H, W]
    """
    device = imgs.device
    # Create a 2D Gaussian kernel
    x = torch.arange(kernel_size) - kernel_size // 2
    x = x.to(device).float()  # Ensure x is on the same device as imgs
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()  # Normalize
    kernel_1d = gauss.unsqueeze(0)  # [1, kernel_size]
    kernel_2d = torch.mm(kernel_1d.T, kernel_1d)  # [kernel_size, kernel_size]
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, kernel_size, kernel_size]

    # Use the kernel for convolution
    channels = imgs.shape[1]
    kernel = kernel_2d.repeat(channels, 1, 1, 1)  # [C, 1, kernel_size, kernel_size]
    padding = kernel_size // 2
    filtered_image = F.conv2d(imgs, kernel, padding=padding, groups=channels)
    return filtered_image

def median_filter(imgs: torch.Tensor, kernel_size: int = 3, differentiable: bool = False) -> torch.Tensor:
    """
    A median filter with differentiable and non-differentiable modes.

    Args:
        imgs (Tensor): Input image tensor of shape [B, C, H, W], Range: [0, 1]
        kernel_size (int): Size of the median filter kernel (must be odd)
        differentiable (bool): If True, use a differentiable approximation of the median filter

    Returns:
        Tensor: Filtered image tensor of shape [B, C, H, W]
    """
    device = imgs.device
    padding = kernel_size // 2
    imgs_padded = F.pad(imgs, (padding, padding, padding, padding), mode='reflect')  # [B, C, H+2*padding, W+2*padding]

    B, C, H, W = imgs.shape
    unfolded = F.unfold(imgs_padded, kernel_size, stride=1)  # [B, C*kernel_size*kernel_size, H*W]
    unfolded = unfolded.view(B, C, kernel_size * kernel_size, H, W)  # [B, C, kernel_size*kernel_size, H, W]

    if differentiable:
        # Use softmax to approximate the median in a differentiable way
        weights = torch.softmax(unfolded, dim=2)  # [B, C, kernel_size*kernel_size, H, W]
        median_approx = (unfolded * weights).sum(dim=2)  # [B, C, H, W]
        return median_approx
    else:
        # Use the median function to compute the median across the kernel dimension
        median = unfolded.median(dim=2).values  # [B, C, H, W]
        return median

def jpeg(imgs:torch.Tensor, quality:int=80, differentiable:bool=False) -> torch.Tensor:
    """
    Differentiable JPEG compression adopted from https://github.com/mlomnitz/DiffJPEG. 

    Args:
        imgs (Tensor): Input image tensor of shape [B, C, H, W], Range: [0, 1]
        quality (int): Quality factor for JPEG compression
        differentiable (bool): If True, use differentiable rounding

    Returns:
        Tensor: Compressed image tensor of shape [B, C, H, W]
    """
    device = imgs.device
    B, C, H, W = imgs.shape
    compressor = DiffJPEG(height=H, width=W, differentiable=differentiable, quality=quality).to(device)
    return compressor.forward(imgs)

def resize(imgs: torch.Tensor, resz_percentage: float, mode: str, differentiable: bool = True) -> torch.Tensor:
    """
    Resize and restore images

    Args:
        imgs (Tensor): Input image tensor of shape [B, C, H, W], Range: [0, 1]
        resz_percentage (float): Percentage to resize the image (e.g., 0.5 for 50%)
        mode (str): Interpolation mode ('bicubic', 'bilinear', etc.)
        differentiable (bool): Placeholder for differentiable flag, not used in this implementation
        
    Returns:
        Tensor: Resized and restored image tensor of shape [B, C, H, W]
    """
    B, C, H, W = imgs.shape
    new_h = int(H * resz_percentage)
    new_w = int(W * resz_percentage)
    resized_imgs = F.interpolate(imgs, size=(new_h, new_w), mode=mode, align_corners=False)
    restored_imgs = F.interpolate(resized_imgs, size=(H, W), mode=mode, align_corners=False)
    return restored_imgs

def compress(imgs: torch.Tensor, method: str, **kwargs: Dict[str, Any]) -> torch.Tensor:
    """
    Compress images using specified method

    Args:
        imgs (Tensor): Input image tensor of shape [B, C, H, W], Range: [0, 1]
        method (str): Compression method ('gaussian', 'median', 'jpeg')
        **kwargs: Additional parameters for the compression method

    Returns:
        Tensor: Compressed image tensor of shape [B, C, H, W]

    Raises:
        ValueError: If the specified method is not supported
    """
    if 'gaussian' in method:
        return gaussian_filter(imgs, **kwargs)
    elif 'median' in method:
        return median_filter(imgs, **kwargs)
    elif 'jpeg' in method:
        return jpeg(imgs, **kwargs)
    elif 'resize' in method:
        return resize(imgs, **kwargs)
    else:
        raise ValueError(f"Unsupported compression method: {method}")
    