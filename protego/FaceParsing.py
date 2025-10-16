import os
from typing import List, Dict, Any
import zipfile

import torch
import torch.nn.functional as F
from torchvision import transforms
import gdown

from .FacialRecognition import download
from FP_DB.FaRL.cfg import pretrain_settings
from . import BASE_PATH

FP_DB_PATH = os.path.join(BASE_PATH, 'FP_DB')
FARL_HOME = os.path.join(FP_DB_PATH, 'FaRL')

BASIC_POOL = ['farl_lapa', 'farl_celebm']

def download(path: str, url: str) -> None:
    """
    A crude download util function, only tailoreed for the scenarios this project meets. 

    Args:
        path (str): The local desired path of the weight file
        url (str): The URL of the file to download.
    """
    folder = "/".join(path.split("/")[:-1])+'/'
    if not os.path.exists(path):
        print(f"Weight {path} does not exist. Downloading it from {url}")
        if "uc?id" in url:
            file_id = url.split("uc?id=")[-1]
            downloaded_f = gdown.download(id=file_id, output=folder, quiet=False)
        elif "view?usp=sharing" in url:
            downloaded_f = gdown.download(url=url, output=folder, fuzzy=True, quiet=False)
        if downloaded_f.endswith(".zip"):
            with zipfile.ZipFile(downloaded_f, 'r') as zip_ref:
                zip_ref.extractall(folder)
            os.remove(downloaded_f)
        elif downloaded_f != path:
            os.rename(downloaded_f, path)
    return

@torch.no_grad()
class FP(object):
    def __init__(self, model_name: str = 'farl_lapa', device: torch.device = torch.device('cpu')):
        self.device = device
        self.model_name = model_name

        if self.model_name == 'farl_lapa':
            self.model = FaRL("lapa/448", device=device)
        elif self.model_name == 'farl_celebm':
            self.model = FaRL("celebm/448", device=device)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

        self.drange = self.model.drange
        self.labels = self.model.labels

    def __call__(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Combine forward and preprocess functions. Do not use .forward() or .preprocess() directly.

        Args:
            img (torch.Tensor): Original Image, shape [1, 3, H, W], range [0, 1], RGB, dtype torch.float32.
            kwargs: additional data required by different models. see specific model preprocess function for details.

        Returns:
            torch.Tensor: Segmentation probabilities, shape [nfaces, nclasses, H, W], range [0, 1]
        """
        return self.model.forward(self.preprocess(img), **kwargs)

    def forward(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward function to detect face and predict segmentation map. Do not use this function directly, use __call__() instead.

        Args:
            img (torch.Tensor): preprocessed face images
            kwargs: additional data required by different models. see specific model forward function for details.

        Returns:
            torch.Tensor: Segmentation probabilities, shape [nfaces, nclasses, H, W], range [0, 1]
        """
        return self.model.forward(img, **kwargs)
    
    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the original image for face parsing. Do not use this function directly, use __call__() instead.

        Args:
            img (torch.Tensor): Original Image, shape [1, 3, H, W], range [0, 1], RGB, dtype torch.float32.

        Returns:
            torch.Tensor: Preprocessed face images
        """
        if img.max() <= 2 and self.drange == 255:
            rescaled_img = img.float() * 255.
        elif img.max() <= 2 and self.drange == 1:
            rescaled_img = img.float()
        elif img.max() > 2 and self.drange == 255:
            rescaled_img = img.float()
        elif img.max() > 2 and self.drange == 1:
            rescaled_img = img.float() / 255.
        return rescaled_img

class FaRL(object):
    def __init__(self, pretrained: str, device = torch.device('cpu')):
        self.device = device
        self.path_dict = {
            'lapa/448': f"{FARL_HOME}/pretrained/face_parsing.farl.lapa.main_ema_136500_jit191.pt",
            'celebm/448': f"{FARL_HOME}/pretrained/face_parsing.farl.celebm.main_ema_181500_jit.pt"
        }
        self.url_dict = {
            'lapa/448': "https://drive.google.com/file/d/1pa4X9-1yvw1A_5N9hJSm9vBAL3zj0xHA/view?usp=sharing",
            'celebm/448': "https://drive.google.com/file/d/10yDTK9jHRBCfRJTtcNlYMzJUMKDlmSxk/view?usp=sharing"
        }
        download(self.path_dict[pretrained], self.url_dict[pretrained])
        self.setting = pretrain_settings[pretrained]
        self.net = torch.jit.load(self.path_dict[pretrained], map_location=self.device).to(self.device).eval()
        self.labels = self.setting['label_names']

        self.drange = 1

    def forward(self, img: torch.Tensor, fd_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Original Image, shape [1, 3, H, W], range [0, 1], RGB, dtype torch.float32.
            fd_data (Dict[str, torch.Tensor]): data from face detection models, requiring:
                - 'rects' (torch.Tensor): Bounding boxes of shape (nfaces, 4).
                - 'points' (torch.Tensor): 5 landmarks of shape (nfaces, 10). (landmark order: left eye, right eye, nose, left mouth corner, right mouth corner)

        Returns:
            torch.Tensor: Segmentation probabilities, shape [nfaces, nclasses, H, W], range [0, 1]
        """
        src_tag = self.setting['matrix_src_tag']
        src_data = fd_data[src_tag]
        if src_tag == "points":
            # Accept either [N, 10] (flattened 5-point landmarks) or [N, 5, 2]
            if src_data.dim() == 2 and src_data.size(1) == 10:
                src_data = src_data.view(-1, 5, 2)
        nfaces = src_data.shape[0]
        H, W = img.shape[-2], img.shape[-1]
        imgs = img.repeat(nfaces, 1, 1, 1)

        matrix = self.setting['get_matrix_fn'](src_data)
        grid = self.setting['get_grid_fn'](matrix=matrix, orig_shape=(H, W))
        inv_grid = self.setting['get_inv_grid_fn'](matrix=matrix, orig_shape=(H, W))
        aligneds = F.grid_sample(img, grid, mode='bilinear', align_corners=False)
        seg_logits, _ = self.net(aligneds)  # nfaces x c x h x w
        seg_logits = F.grid_sample(seg_logits, inv_grid, mode='bilinear', align_corners=False)
        seg_probs = torch.softmax(seg_logits, dim=1)
        return seg_probs  # nfaces x nclasses x h x w
