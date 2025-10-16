import os
from typing import List, Dict
import zipfile

import torch
import torch.nn.functional as F
from torchvision import transforms
import gdown

from FAttri_DB.FaRL.cfg import pretrain_settings
from FAttri_DB.FaRL.net import farl_classification
from . import BASE_PATH

FAttri_DB_PATH = os.path.join(BASE_PATH, 'FAttri_DB')
FARL_HOME = os.path.join(FAttri_DB_PATH, 'FaRL')

BASIC_POOL = ['farl_celeba224']

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
class FAttri(object):
    def __init__(self, model_name: str = 'farl_celeba224', device: torch.device = torch.device('cpu')):
        self.device = device
        self.model_name = model_name

        if self.model_name == 'farl_celeba224':
            self.model = FaRL("celeba/224", device=device)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

        self.drange = self.model.drange
        self.labels = self.model.labels

    def __call__(self, img: torch.Tensor, **kwargs) -> List[Dict[str, float]]:
        """
        Combine forward and preprocess functions.

        Args:
            img (torch.Tensor): Original Image, shape [1, 3, H, W], range [0, 1], RGB, dtype torch.float32.
            kwargs: additional data required by different models. see specific model preprocess function for details.

        Returns:
            List[Dict[str, float]]: List of attributes' probability for each detected face.
        """
        img = self.preprocess(img, **kwargs)
        return self.forward(img)

    def forward(self, img: torch.Tensor) -> List[Dict[str, float]]:
        """
        Forward function to predict attributes.

        Args:
            img (torch.Tensor): preprocessed face images
        Returns:
            List[Dict[str, float]]: List of attributes' probability for each detected face.
        """
        return self.model.forward(img)

    def preprocess(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Preprocess the cropped face image for attribute prediction.

        Args:
            img (torch.Tensor): Original Image, shape [1, 3, H, W], range [0, 1], RGB, dtype torch.float32.
            kwargs: additional data required by different models. see specific model preprocess function for details.
        
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
        return self.model.preprocess(rescaled_img, **kwargs)

class FaRL(object):
    def __init__(self, pretrained: str, device: torch.device = torch.device('cpu')) -> None:
        self.device = device
        self.path_dict = {
            'celeba/224': f"{FARL_HOME}/pretrained/face_attribute.farl.celeba.pt"
        }
        self.url_dict = {
            'celeba/224': "https://drive.google.com/file/d/1yo7BVO4RMeq-5fCmuk_VxvNL06zIEh-C/view?usp=sharing"
        }
        download(self.path_dict[pretrained], self.url_dict[pretrained])
        self.setting = pretrain_settings[pretrained]
        self.labels = self.setting["classes"]
        self.net = farl_classification(num_classes=self.setting["num_classes"], layers=self.setting["layers"])
        self.net.load_state_dict(torch.load(self.path_dict[pretrained], map_location=device, weights_only=False))
        self.net.to(device).eval()
        
        self.drange = 1

    def forward(self, img: torch.Tensor) -> List[Dict[str, float]]:
        """
        Forward function to detect face and predict attributes.

        Args:
            img (torch.Tensor): preprocessed face images, shape [nfaces, 3, 224, 224], range [0, 1], RGB, dtype torch.float32.
        Returns:
            List[Dict[str, float]]: List of attributes' probability for each detected face in the same order as input.
        """
        probs = torch.sigmoid(self.net(img))
        attris = []
        for i in range(probs.shape[0]):
            attri = {}
            for j, label in enumerate(self.labels):
                attri[label] = probs[i, j].item()
            attris.append(attri)
        return attris

    def preprocess(self, img: torch.Tensor, fd_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess the cropped face image for attribute prediction.

        Args:
            img (torch.Tensor): Original Image, shape [1, 3, H, W], range [0, 1], RGB, dtype torch.float32.
            fd_data (Dict[str, torch.Tensor]): data from face detection models, requiring:
                - 'rects' (torch.Tensor): Bounding boxes of shape (nfaces, 4).
                - 'points' (torch.Tensor): 5 landmarks of shape (nfaces, 10). (landmark order: left eye, right eye, nose, left mouth corner, right mouth corner)
        Returns:
            torch.Tensor: Preprocessed face images, shape [nfaces, 3, 224, 224], range [0, 1], RGB, dtype torch.float32.
        """
        # Determine number of faces from detection data to keep grid/image batches aligned
        src_tag = self.setting["matrix_src_tag"]
        src_data = fd_data[src_tag]
        if src_tag == "points":
            # Accept either [N, 10] (flattened 5-point landmarks) or [N, 5, 2]
            if src_data.dim() == 2 and src_data.size(1) == 10:
                src_data = src_data.view(-1, 5, 2)
        # Derive face count from src_data
        nfaces = src_data.shape[0]

        # Repeat the input image per face
        H, W = img.shape[-2], img.shape[-1]
        imgs = img.repeat(nfaces, 1, 1, 1)

        # Build alignment matrix and sampling grid
        matrix = self.setting["get_matrix_fn"](src_data)
        grid = self.setting["get_grid_fn"](matrix=matrix, orig_shape=(H, W))
        aligneds = F.grid_sample(imgs, grid, mode='bilinear', align_corners=False)
        preprocess = transforms.Compose([
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        return torch.stack([preprocess(aligneds[i]) for i in range(nfaces)], dim=0)
