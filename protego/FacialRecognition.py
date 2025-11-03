import os
import copy
import shutil
from typing import List, Any, Union, Optional, Dict
from collections import OrderedDict
import zipfile
import datetime

import torch
import torch.nn.functional as F
from torchvision import transforms
import kornia
import gdown
import numpy as np
import cv2
from skimage import transform as skitrans

from FR_DB.ir50_opom.IR import IR_50
from FR_DB.arcface.arcface import Arcface
from FR_DB.adaface import net
from FR_DB.facenet.inception_resnet_v1 import InceptionResnetV1
from FR_DB.magface import iresnet
from FR_DB.magface.iresnet import IResNet
from FR_DB.vit.vit_face import ViT_face
from FR_DB.vit.vits_face import ViTs_face
from FR_DB.part_fvit.part_fvit import ViT_face_landmark_patch8
from FR_DB.transface import get_model as get_transface_model
from FR_DB.swinface import build_model as build_swinface_model
from FR_DB.facevit.vit_face_model import ViT_face_model as CrossImage_Hybrid_ViT
from FR_DB.facevit.vit_face_model import Hybrid_ViT
from FR_DB.facevit.resnet import resnet_face18
from .FaceDetection import FD
from . import BASE_PATH

FR_DB_PATH = os.path.join(BASE_PATH, 'FR_DB')
IR50_OPOM_HOME = os.path.join(FR_DB_PATH, 'ir50_opom')
ADAFACE_HOME = os.path.join(FR_DB_PATH, 'adaface')
ARCFACE_HOME = os.path.join(FR_DB_PATH, 'arcface')
FACENET_HOME = os.path.join(FR_DB_PATH, 'facenet')
MAGFACE_HOME = os.path.join(FR_DB_PATH, 'magface')
VIT_HOME = os.path.join(FR_DB_PATH, 'vit')
PARTFVIT_HOME = os.path.join(FR_DB_PATH, 'part_fvit')
TRANSFACE_HOME = os.path.join(FR_DB_PATH, 'transface')
SWINFACE_HOME = os.path.join(FR_DB_PATH, 'swinface')
FACEVIT_HOME = os.path.join(FR_DB_PATH, 'facevit')

SPECIAL_POOL = ['inception_facenet_vgg', 'inception_facenet_casia',
              'ir50_magface_ms1mv2', 'ir100_magface_ms1mv2', 
              'vit_cosface_ms1mv2', 'vits_cosface_ms1mv2', 
              'partfvit_cosface_nosl_webface', 'partfvit_cosface_nosl_ms1mv3', 
              'transfaces_arcface_ms1mv2', 'transfaceb_arcface_ms1mv2', 'transfacel_arcface_ms1mv2', 'transfaces_arcface_glint360k', 'transfaceb_arcface_glint360k', 'transfacel_arcface_glint360k',
              'swinfacet_arcface_ms1mv2', 
              'facevit_arcface_singled8h1_webface', 'facevit_arcface_crossd8h1_webface', 'facevit_arcface_crossd8h2_webface']
BASIC_POOL = ['ir50_softmax_casia', 'ir50_cosface_casia', 
              'ir50_arcface_casia', 'mobilenet_arcface_casia', 'mobilefacenet_arcface_casia', 
              'ir18_adaface_webface', 'ir50_adaface_ms1mv2', 'ir50_adaface_casia', 'ir50_adaface_webface', 'ir101_adaface_webface']
VIT_FAMILY = SPECIAL_POOL[4:]
FINETUNE_POOL = ['ir50_adaface_fsorig', 'ir50_adaface_fsclean', 'ir50_adaface_fsprot20', 
                 'ir50_adaface_fsprot50', 'ir50_adaface_fsprot80', 'ir50_adaface_fsprot5014frs', 
                 'ir50_adaface_fsprot50nokmeans']

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
        elif "usp=sharing" in url or "usp=share_link" in url:
            if "file" in url:
                downloaded_f = gdown.download(url=url, output=folder, fuzzy=True, quiet=False)
                if downloaded_f.endswith(".zip"):
                    with zipfile.ZipFile(downloaded_f, 'r') as zip_ref:
                        zip_ref.extractall(folder)
                    os.remove(downloaded_f)
                elif downloaded_f != path:
                    os.rename(downloaded_f, path)
            elif "folder" in url:
                download_fs = gdown.download_folder(url=url, output=folder, quiet=False)
                for f in download_fs:
                    shutil.move(f, os.path.join(folder, f.split("/")[-1]))
                os.removedirs(os.path.join(folder, download_fs[0].split("/")[-2]))
            else:
                raise ValueError(f"Unrecognized URL format: {url}")
    return
    
class FR(object):
    """
    Attributes:
        device (torch.device): The device to run the model on.
        model_name (str): The name of the FR model.
        fr_model (object): The actual FR model object.
        embedding_dim (int): The dimension of the feature vector.
        dis_func (str): The distance function used during training. 
        img_size (int): The expected size of the input image.
        drange (int): The range of pixel values in the expected input image.
        preprocess_method (callable): The preprocessing method for the input image.
        l2norm (bool): Whether the direct output of the model is L2-normalized.

    Methods:
        __init__(model_name: str, device: torch.device):
            Initializes the FR model with the specified model name and device.
        __call__(imgs: torch.Tensor) -> torch.Tensor:
            Combines the preprocessing and feature extraction steps.
        extract_features(images: torch.Tensor) -> torch.Tensor:
            Extracts features from a batch of images.
        preprocess(images: torch.Tensor) -> torch.Tensor:
            Preprocesses a batch of images for the FR model.
    """
    def __init__(self, model_name: str = 'inception_facenet_vgg', device: torch.device = torch.device('cpu')):
        """
        Init the FR model with the specified model name and device.

        Args:
            model_name (str): The name of the FR model. Default is 'inception_facenet_vgg'. Options are:
            (- 'model_name': Neural Structure + Loss + Dataset)
            - 'inception_facenet_vgg': InceptionResnetV1 + Facenet + VGGFace2
            - 'inception_facenet_casia': InceptionResnetV1 + Facenet + CASIA-WebFace
            - 'ir50_softmax_casia': Improved ResNet50 + Softmax + CASIA-WebFace
            - 'ir50_cosface_casia': Improved ResNet50 + CosFace + CASIA-WebFace
            - 'ir50_arcface_casia': Improved ResNet50 + ArcFace + CASIA-WebFace
            - 'ir50_adaface_casia': Improved ResNet50 + AdaFace + CASIA-WebFace
            - 'mobilenet_arcface_casia': MobileNet + ArcFace + CASIA-WebFace
            - 'mobilefacenet_arcface_casia': MobileFacenet + ArcFace + CASIA-WebFace
            - 'ir18_adaface_webface': Improved ResNet18 + AdaFace + WebFace4M
            - 'ir50_adaface_webface': Improved ResNet50 + AdaFace + WebFace4M
            - 'ir101_adaface_webface': Improved ResNet100 + AdaFace + WebFace4M
            - 'ir50_adaface_ms1mv2': Improved ResNet50 + AdaFace + MS1MV2
            - 'inception_facenet_vgg': InceptionResnetV1 + Facenet + VGGFace2
            - 'inception_facenet_casia': InceptionResnetV1 + Facenet + CASIA-WebFace
            - 'ir50_magface_ms1mv2': Improved ResNet50 + MagFace + MS1MV2
            - 'ir100_magface_ms1mv2': Improved ResNet100 + MagFace + MS1MV2
            device (torch.device): The device to run the model on. Default is CPU.

        Raises:
            ValueError: If the model name is not valid.

        Example:
            >>> import torch
            >>> from FacialRecognition import * 
            >>> model_name = BASIC_POOL[0] # Choose a model from the model pool
            >>> fr = FR(model_name=model_name, device=torch.device('cpu'))
            >>> img = torch.randint(0, 255, (1, 3, 224, 224), dtype=torch.uint8) # Shape: (B, 3, H, H), Range: [0, 255], RGB, dtype: torch.uint8
            >>> img = img.float() # Convert to float32
            >>> features = fr(img) 
            >>> print(features.shape) # Should be (1, 512) for most models. For 'mobilefacenet_arcface_casia', it should be (1, 128)
        """
        self.device = device
        self.model_name = model_name

        if self.model_name == 'ir50_softmax_casia': # Improved ResNet50 + Softmax + CASIA-WebFace
            self.fr_model = IR50_OPOM(loss='softmax', device=self.device)
        elif self.model_name == 'ir50_cosface_casia': # Improved ResNet50 + CosFace + CASIA-WebFace
            self.fr_model = IR50_OPOM(loss='cosface', device=self.device)
        elif self.model_name == 'ir50_arcface_casia': # Improved ResNet50 + ArcFace + CASIA-WebFace
            self.fr_model = ArcFace(backbone='iresnet50', device=self.device)
        elif self.model_name == 'ir50_adaface_casia': # Improved ResNet50 + AdaFace + CASIA-WebFace
            self.fr_model = AdaFace(backbone='ir_50_casia', device=self.device)
        elif self.model_name == 'mobilenet_arcface_casia': # MobileNet + ArcFace + CASIA-WebFace
            self.fr_model = ArcFace(backbone='mobilenetv1', device=self.device)
        elif self.model_name == 'mobilefacenet_arcface_casia': # MobileFacenet + ArcFace + CASIA-WebFace
            self.fr_model = ArcFace(backbone='mobilefacenet', device=self.device)
        elif self.model_name == 'ir18_adaface_webface': # Improved ResNet18 + AdaFace + WebFace4M
            self.fr_model = AdaFace(backbone='ir_18_web', device=self.device)
        elif self.model_name == 'ir50_adaface_webface': # Improved ResNet50 + AdaFace + WebFace4M
            self.fr_model = AdaFace(backbone='ir_50_web', device=self.device)
        elif self.model_name == 'ir101_adaface_webface': # Improved ResNet101 + AdaFace + WebFace4M
            self.fr_model = AdaFace(backbone='ir_101_web', device=self.device)
        elif self.model_name == 'ir50_adaface_ms1mv2': # Improved ResNet50 + AdaFace + MS1MV2
            self.fr_model = AdaFace(backbone='ir_50_ms1mv2', device=self.device)
        elif self.model_name == 'inception_facenet_vgg': # InceptionResnetV1 + Facenet + VGGFace2
            self.fr_model = InceptionFacenet(pretrained='vggface2', device=self.device)
        elif self.model_name == 'inception_facenet_casia': # InceptionResnetV1 + Facenet + CASIA-WebFace
            self.fr_model = InceptionFacenet(pretrained='casia-webface', device=self.device)
        elif self.model_name == 'ir50_magface_ms1mv2': # Improved ResNet50 + MagFace + MS1MV2
            self.fr_model = Magface(arch='iresnet50', device=self.device)
        elif self.model_name == 'ir100_magface_ms1mv2': # Improved ResNet100 + MagFace + MS1MV2
            self.fr_model = Magface(arch='iresnet100', device=self.device)
        elif self.model_name == 'vit_cosface_ms1mv2': # ViT + CosFace + MS1MV2
            self.fr_model = ViT(backbone='vit', device=self.device)
        elif self.model_name == 'vits_cosface_ms1mv2': # ViTs + CosFace + MS1MV2
            self.fr_model = ViT(backbone='vits', device=self.device)
        elif self.model_name == 'partfvit_cosface_nosl_webface': # Part-based fViT + CosFace + WebFace4M (No pretrain, Retrained with Supervised Learning)
            self.fr_model = PartfViT(model_name=model_name, device=self.device)
        elif self.model_name == 'partfvit_cosface_nosl_ms1mv3': # Part-based fViT + CosFace + MS1MV3 (No pretrain, Retrained with Supervised Learning)
            self.fr_model = PartfViT(model_name=model_name, device=self.device)
        elif self.model_name == 'partfvit_cosface_lafssl_webface': # Part-based fViT + CosFace + WebFace4M (LAFS pretrain, Retrained with Supervised Learning)
            self.fr_model = PartfViT(model_name=model_name, device=self.device)
        elif self.model_name == 'fvit_cosface_dinossl_webface': # fViT + CosFace + WebFace4M (Self-supervised learning with DINO)
            self.fr_model = PartfViT(model_name=model_name, device=self.device)
        elif self.model_name == 'partfvit_cosface_lafsssl_webface': # Part-based fViT + CosFace + WebFace4M (Self-supervised learning with LAFS)
            self.fr_model = PartfViT(model_name=model_name, device=self.device)
        elif self.model_name == 'transfaces_arcface_ms1mv2': # TransFace-S + ArcFace + MS1MV2
            self.fr_model = TransFace(size='s', pretrained='ms1mv2', device=self.device)
        elif self.model_name == 'transfaceb_arcface_ms1mv2': # TransFace-B + ArcFace + MS1MV2
            self.fr_model = TransFace(size='b', pretrained='ms1mv2', device=self.device)
        elif self.model_name == 'transfacel_arcface_ms1mv2': # TransFace-L + ArcFace + MS1MV2
            self.fr_model = TransFace(size='l', pretrained='ms1mv2', device=self.device)
        elif self.model_name == 'transfaces_arcface_glint360k': # TransFace-S + ArcFace + Glint360k
            self.fr_model = TransFace(size='s', pretrained='glint360k', device=self.device)
        elif self.model_name == 'transfaceb_arcface_glint360k': # TransFace-B + ArcFace + Glint360k
            self.fr_model = TransFace(size='b', pretrained='glint360k', device=self.device)
        elif self.model_name == 'transfacel_arcface_glint360k': # TransFace-L + ArcFace + Glint360k
            self.fr_model = TransFace(size='l', pretrained='glint360k', device=self.device)
        elif self.model_name == 'swinfacet_arcface_ms1mv2': # SwinFace-T + ArcFace + MS1MV2
            self.fr_model = SwinFace(model_name='swin_t', device=self.device)
        elif self.model_name == 'facevit_arcface_singled8h1_webface': # FaceViT (Single-image attention, depth=8, heads=1) + ArcFace + WebFace2M
            self.fr_model = FaceViT(cross_image_attention=False, depth=8, heads=1, device=self.device)
        elif self.model_name == 'facevit_arcface_singled8h2_webface': # FaceViT (Single-image attention, depth=8, heads=2) + ArcFace + WebFace2M
            self.fr_model = FaceViT(cross_image_attention=False, depth=8, heads=2, device=self.device)
        elif self.model_name == 'facevit_arcface_crossd8h1_webface': # FaceViT (Cross-image attention, depth=8, heads=1) + ArcFace + WebFace2M
            self.fr_model = FaceViT(cross_image_attention=True, depth=8, heads=1, device=self.device)
        elif self.model_name == 'facevit_arcface_crossd8h2_webface': # FaceViT (Cross-image attention, depth=8, heads=2) + ArcFace + WebFace2M
            self.fr_model = FaceViT(cross_image_attention=True, depth=8, heads=2, device=self.device)

        elif self.model_name == 'ir50_adaface_fsorig': # Improved ResNet50 + AdaFace + FSOrig
            self.fr_model = AdaFace(backbone='ir_50_fsorig', device=self.device)
        elif self.model_name == 'ir50_adaface_fsclean': # Improved ResNet50 + AdaFace + FSClean
            self.fr_model = AdaFace(backbone='ir_50_fsclean', device=self.device)
        elif self.model_name == 'ir50_adaface_fsprot20': # Improved ResNet50 + AdaFace + FSProt20
            self.fr_model = AdaFace(backbone='ir_50_fsprot20', device=self.device)
        elif self.model_name == 'ir50_adaface_fsprot50': # Improved ResNet50 + AdaFace + FSProt50
            self.fr_model = AdaFace(backbone='ir_50_fsprot50', device=self.device)
        elif self.model_name == 'ir50_adaface_fsprot80': # Improved ResNet50 + AdaFace + FSProt80
            self.fr_model = AdaFace(backbone='ir_50_fsprot80', device=self.device)
        elif self.model_name == 'ir50_adaface_fsprot5014frs': # Improved ResNet50 + AdaFace + FSProt50_14FRs
            self.fr_model = AdaFace(backbone='ir_50_fsprot5014frs', device=self.device)
        elif self.model_name == 'ir50_adaface_fsprot50nokmeans': # Improved ResNet50 + AdaFace + FSProt50NoKmeans
            self.fr_model = AdaFace(backbone='ir_50_fsprot50nokmeans', device=self.device)
        else:
            raise ValueError(f"Invalid model name: {self.model_name}")
        
        self.embedding_dim = self.fr_model.embedding_dim
        self.dis_func = self.fr_model.dis_func
        self.img_size = self.fr_model.img_size
        self.drange = self.fr_model.drange
        self.preprocess_method = self.fr_model.preprocessing
        self.l2norm = self.fr_model.l2norm

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Combines the preprocessing and feature extraction steps.

        Args:
            images (torch.Tensor): A batch of images with shape (B, 3, H, H), dtype: float32, RGB format.

        Returns:
            torch.Tensor: Features with shape (B, C). Refer to the `extract_features` method for details.
        """
        return self.extract_features(self.preprocess(imgs))

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from a batch of images.

        Args:
            images (torch.Tensor): A batch of preprocessed images with shape (B, 3, H, W).

        Returns:
            torch.Tensor: Features with shape (B, C). 
                - 'ir50_softmax_casia' : (B, 512)
                - 'ir50_cosface_casia' : (B, 512)
                - 'ir50_arcface_casia' : (B, 512)
                - 'ir50_adaface_casia' : (B, 512)
                - 'mobilenet_arcface_casia' : (B, 512)
                - 'mobilefacenet_arcface_casia' : (B, 128)
                - 'ir18_adaface_webface' : (B, 512)
                - 'ir50_adaface_webface' : (B, 512)
                - 'ir101_adaface_webface' : (B, 512)
                - 'ir50_adaface_ms1mv2' : (B, 512)
                - 'inception_facenet_vgg' : (B, 512)
                - 'inception_facenet_casia' : (B, 512)
        """
        return self.fr_model.forward(images)
    
    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses a batch of images for the FR model.

        Args:
            images (torch.Tensor): A batch of images with shape (B, 3, H, H), dtype: float32, RGB format.

        Returns:
            torch.Tensor: Preprocessed images with shape (B, 3, H', H'). Range [0, 1] or [-1, 1] or [0, 255], dtype: float32., RGB or BGR format.
        """
        if images.max() >= 2 and self.drange == 1:
            rescaled_images = images.float() / 255.
        elif images.max() >= 2 and self.drange == 255:
            rescaled_images = images
        elif images.max() < 2 and self.drange == 1:
            rescaled_images = images
        elif images.max() < 2 and self.drange == 255:
            rescaled_images = images.float() * 255.
        preprocessed = torch.stack([self.preprocess_method(img) for img in rescaled_images])
        return preprocessed
    
class Magface(object):
    """
    Model: IResNet50/IResNet100
    Datasets: MS1MV2
    Loss Function: MagFace
    Source: https://github.com/IrvingMeng/MagFace

    Args:
        arch (str): Backbone model name. Options are 'iresnet50', 'iresnet100'.
        device (torch.device): The device to run the model on.
    """
    def __init__(self, arch: str, device: torch.device):
        self.embedding_dim = 512
        self.path_dict = {
            'iresnet100': os.path.join(MAGFACE_HOME, 'pretrained/magface_iresnet100_MS1MV2_ddp.pth'),
            'iresnet50': os.path.join(MAGFACE_HOME, 'pretrained/magface_iresnet50_MS1MV2_ddp_fp32.pth')
        }
        self.url_dict = {
            'iresnet100': "https://drive.google.com/file/d/1Bd87admxOZvbIOAyTkGEntsEz3fyMt7H/view?usp=sharing",
            'iresnet50': "https://drive.google.com/file/d/1QPNOviu_A8YDk9Rxe8hgMIXvDKzh6JMG/view?usp=sharing"
        }
        download(self.path_dict[arch], self.url_dict[arch])
        self.model = MagfaceNetworkBuilder(arch=arch, embedding_size=self.embedding_dim, weight_path=self.path_dict[arch], device=device)
        self.model = self.model.to(device)
        self.model = self.model.eval() 

        self.preprocessing = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),
            transforms.Lambda(lambda x: x[[2, 1, 0], :, :]) # ! Convert RGB to BGR for MagFace
        ])
        self.img_size = 112 
        self.drange = 1 
        self.dis_func = "cosine" 
        self.l2norm = False

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.model.forward(imgs)
    
class MagfaceNetworkBuilder(torch.nn.Module):
    def __init__(self, arch:str, embedding_size:int, weight_path: str, device: torch.device):
        super(MagfaceNetworkBuilder, self).__init__()
        self.model = self.get_net(arch, embedding_size)
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        self.load_weights(weight_path, device)

    def get_net(self, arch:str, embedding_size:int) -> IResNet:
        if arch == 'iresnet34':
            return iresnet.iresnet34(pretrained=False, num_classes=embedding_size,)
        elif arch == 'iresnet18':
            return iresnet.iresnet18(pretrained=False, num_classes=embedding_size,)
        elif arch == 'iresnet50':
            return iresnet.iresnet50(pretrained=False, num_classes=embedding_size,)
        elif arch == 'iresnet100':
            return iresnet.iresnet100(pretrained=False, num_classes=embedding_size,)
        else:
            raise ValueError(f"Unknown architecture for Magface: {arch}")
        
    def load_weights(self, weight_path: str, device: torch.device) -> None:
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        ckpts = torch.load(weight_path, map_location=device, weights_only=False) # ! Note that the older torch.load function set `weights_only=False` by default, so we need to set it explicitly
        #print(ckpts['state_dict'].keys())
        #print(self.model.state_dict().keys())
        state_dict = self.clean_dict_inf(ckpts['state_dict'])
        model_dict = self.model.state_dict()
        model_dict.update(state_dict)
        self.model.load_state_dict(model_dict)
    
    def clean_dict_inf(self, state_dict: OrderedDict) -> OrderedDict:
        _state_dict = OrderedDict()
        for k, v in state_dict.items():
            # # assert k[0:1] == 'features.module.'
            #new_k = 'features.'+'.'.join(k.split('.')[2:])
            new_k = '.'.join(k.split('.')[2:]) # ! Different from the original mapping due to different load method
            if new_k in self.model.state_dict().keys() and v.size() == self.model.state_dict()[new_k].size():
                _state_dict[new_k] = v
            # assert k[0:1] == 'module.features.'
            new_kk = '.'.join(k.split('.')[1:])
            if new_kk in self.model.state_dict().keys() and v.size() == self.model.state_dict()[new_kk].size():
                _state_dict[new_kk] = v
        num_model = len(self.model.state_dict().keys())
        num_ckpt = len(_state_dict.keys())
        if num_model != num_ckpt:
            raise ValueError(f"Model state dict size {num_model} does not match checkpoint state dict size {num_ckpt}.")
        return _state_dict
        
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.model(imgs)

class InceptionFacenet(object):
    """
    Model: InceptionResnetV1
    Datasets: VGGFace2/CASIA-WebFace
    Loss Function: Facenet
    Source: https://github.com/timesler/facenet-pytorch

    Args:
        pretrained (str): Pretrained model name. Options are 'casia-webface', 'vggface2'.
        device (torch.device): The device to run the model on.
    """
    def __init__(self, pretrained: str, device: torch.device):
        self.model = InceptionResnetV1(classify=False, pretrained=pretrained).to(device)
        self.model = self.model.eval()

        self.preprocessing = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.Lambda(lambda x: x * 255.0),
            transforms.Lambda(lambda x: (x - 127.5) / 128.) # ! Same as the original normalization in facenet-pytorch
        ])
        self.img_size = 160
        self.drange = 1
        self.dis_func = "euclidean"
        self.embedding_dim = 512
        self.l2norm = True

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.model(imgs)

class IR50_OPOM(object):
    """
    Model: Improved ResNet50
    Datasets: CASIA-WebFace
    Loss Function: Softmax/CosFace
    Source: https://github.com/zhongyy/OPOM

    Args:
        weight_path (str): Path to the model weights.
        device (torch.device): The device to run the model on.
    """
    def __init__(self, loss: str, device: torch.device):
        self.path_dict = {
            'softmax': os.path.join(IR50_OPOM_HOME, 'pretrained/Backbone_IR_50_Epoch_94_Batch_180000_Time_2020-03-30-01-55_checkpoint.pth'),
            'cosface': os.path.join(IR50_OPOM_HOME, 'pretrained/Backbone_IR_50_Epoch_74_Batch_140000_Time_2020-05-27-21-38_checkpoint.pth')
        }
        self.url_dict = {
            'softmax': "https://drive.google.com/file/d/10YKGalAc3B0ywudFMUp61HAMij8p9xhW/view?usp=sharing",
            'cosface': "https://drive.google.com/file/d/1NUm9jYoC-q0f2Yz8rYky8NdgwRgu4-vr/view?usp=sharing"
        }
        download(self.path_dict[loss], self.url_dict[loss])
        weight_path = self.path_dict[loss]
        self.model = IR_50([112, 112])
        self.model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=False))
        self.model = self.model.to(device)
        self.model = self.model.eval()

        self.preprocessing = transforms.Compose([
            transforms.Resize((112, 112))
            #transforms.Lambda(lambda x: x * 255.0)
        ])
        self.img_size = 112
        self.drange = 255
        self.dis_func = "cosine" if loss == 'cosface' else 'euclidean'
        self.embedding_dim = 512 
        self.l2norm = False 

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.model.forward(imgs)
    
class ArcFace(object):
    """
    Model: IR50/MobileNetV1/MobileFacenet
    Datasets: CASIA-WebFace
    Loss Function: ArcFace
    Source: https://github.com/bubbliiiing/arcface-pytorch

    Args:
        backbone (str): Backbone model name. Options are 'iresnet50', 'mobilenetv1', 'mobilefacenet'.
        device (torch.device): The device to run the model on.
    """
    def __init__(self, backbone: str, device: torch.device):
        self.path_dict = {
            'iresnet50': os.path.join(ARCFACE_HOME, 'pretrained/arcface_iresnet50.pth'),
            'mobilenetv1': os.path.join(ARCFACE_HOME, 'pretrained/arcface_mobilenet_v1.pth'),
            'mobilefacenet': os.path.join(ARCFACE_HOME, 'pretrained/arcface_mobilefacenet.pth')
        }
        self.url_dict = {
            'iresnet50': "https://drive.google.com/file/d/1V-dNjoNaXrEVlt5G9DIP5kmIjo5s9Pbk/view?usp=sharing",
            'mobilenetv1': "https://drive.google.com/file/d/13qtdGZc7YmovAnk1Sf0K5AEG8fiCu_nC/view?usp=sharing", 
            'mobilefacenet': "https://drive.google.com/file/d/1YZN0K-eEB3IF-HDZdXfXR-ztxTMPnYq2/view?usp=sharing"
        }
        download(self.path_dict[backbone], self.url_dict[backbone])
        model_path = self.path_dict[backbone]

        self.model = Arcface(backbone=backbone, mode='predict')
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False), strict=False)
        self.model = self.model.to(device)
        self.model = self.model.eval()

        self.preprocessing = transforms.Compose([
            transforms.Resize((112, 112), interpolation=transforms.InterpolationMode.BICUBIC), # ! Same as the original ArcFace preprocessing
            transforms.Lambda(lambda x: (x - 0.5) / 0.5)
        ])
        self.img_size = 112
        self.drange = 1
        self.dis_func = "cosine"
        self.embedding_dim = 512 if not backbone == 'mobilefacenet' else 128
        self.l2norm = True

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.model.forward(imgs)
    
class AdaFace(object):
    """
    Model: Improved ResNet18/50/101
    Datasets: WebFace4M/CASIA-WebFace/MS1MV2
    Loss Function: AdaFace
    Source: https://github.com/mk-minchul/AdaFace

    Args:
        backbone (str): Backbone model name. Options are 'ir_18_web', 'ir_50_web', 'ir_50_casia', 'ir_101_web', 'ir_50_ms1mv2'.
        device (torch.device): The device to run the model on.
    """
    def __init__(self, backbone: str, device: torch.device):
        self.path_dict = {
            'ir_18_web': os.path.join(ADAFACE_HOME, 'pretrained/adaface_ir18_webface4m.ckpt'),
            'ir_50_casia': os.path.join(ADAFACE_HOME, 'pretrained/adaface_ir50_casia.ckpt'),
            'ir_50_web': os.path.join(ADAFACE_HOME, 'pretrained/adaface_ir50_webface4m.ckpt'),
            'ir_101_web': os.path.join(ADAFACE_HOME, 'pretrained/adaface_ir101_webface4m.ckpt'),
            'ir_50_ms1mv2': os.path.join(ADAFACE_HOME, 'pretrained/adaface_ir50_ms1mv2.ckpt'), 
            'ir_50_fsorig': os.path.join(ADAFACE_HOME, 'pretrained/adaface_ir50_fsorig.ckpt'),
            'ir_50_fsclean': os.path.join(ADAFACE_HOME, 'pretrained/adaface_ir50_fsclean.ckpt'), 
            'ir_50_fsprot20': os.path.join(ADAFACE_HOME, 'pretrained/adaface_ir50_fsprot20.ckpt'),
            'ir_50_fsprot50': os.path.join(ADAFACE_HOME, 'pretrained/adaface_ir50_fsprot50.ckpt'),
            'ir_50_fsprot80': os.path.join(ADAFACE_HOME, 'pretrained/adaface_ir50_fsprot80.ckpt'),
            'ir_50_fsprot5014frs': os.path.join(ADAFACE_HOME, 'pretrained/adaface_ir50_fsprot5014frs.ckpt'), 
            'ir_50_fsprot50nokmeans': os.path.join(ADAFACE_HOME, 'pretrained/adaface_ir50_fsprot50_nokmeans.ckpt')
        }
        self.url_dict = {
            'ir_18_web': "https://drive.google.com/file/d/1J17_QW1Oq00EhSWObISnhWEYr2NNrg2y/view?usp=sharing",
            'ir_50_casia': "https://drive.google.com/file/d/1g1qdg7_HSzkue7_VrW64fnWuHl0YL2C2/view?usp=sharing",
            'ir_50_web': "https://drive.google.com/file/d/1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN/view?usp=sharing",
            'ir_101_web': "https://drive.google.com/file/d/18jQkqB0avFqWa0Pas52g54xNshUOQJpQ/view?usp=sharing",
            'ir_50_ms1mv2': "https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing"
        }
        if backbone in self.url_dict.keys():
            download(self.path_dict[backbone], self.url_dict[backbone])
        model_path = self.path_dict[backbone]
        self.model = net.build_model("_".join(backbone.split('_')[:2]))
        statedict = torch.load(model_path, weights_only=False, map_location=device)['state_dict']
        model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
        self.model.load_state_dict(model_statedict)
        self.model = self.model.to(device)
        self.model = self.model.eval()

        self.preprocessing = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.Lambda(lambda x: (x - 0.5) / 0.5), 
            transforms.Lambda(lambda x: x[[2, 1, 0], :, :])
        ])
        self.img_size = 112
        self.drange = 1
        self.dis_func = "cosine"
        self.embedding_dim = 512 
        self.l2norm = True 

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.model(imgs)[0]

class ViT(object):
    """
    Model: ViT/ViTs (Pure and simplest ViT. ViT does hard patching while ViTs does overlapping/soft patching)
    Datasets: MS1MV2
    Loss Function: CosFace
    Source: https://github.com/zhongyy/Face-Transformer (arXiv)

    Args:
        backbone (str): Backbone model name. Options are 'vit', 'vits'.
        device (torch.device): The device to run the model on.
    """
    def __init__(self, backbone: str, device: torch.device):
        self.path_dict = {
            'vit': os.path.join(VIT_HOME, 'pretrained/Backbone_VIT_Epoch_2_Batch_20000_Time_2021-01-12-16-48_checkpoint.pth'),
            'vits': os.path.join(VIT_HOME, 'pretrained/Backbone_VITs_Epoch_2_Batch_12000_Time_2021-03-17-04-05_checkpoint.pth')
        }
        self.url_dict = {
            'vit': "https://drive.google.com/file/d/1OZRU430CjABSJtXU0oHZHlxgzXn6Gaqu/view?usp=share_link", 
            'vits': "https://drive.google.com/file/d/1U7c_ojiuRPBfolvziB_VthksABHaFKud/view?usp=share_link"
        }
        download(self.path_dict[backbone], self.url_dict[backbone])
        model_path = self.path_dict[backbone]
        if backbone == 'vit':
            self.model = ViT_face(
                image_size=112, 
                patch_size=8, 
                loss_type='CosFace', 
                GPU_ID=device, 
                num_class=93431, 
                dim=512, 
                depth=20, 
                heads=8, 
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1)
        elif backbone == 'vits':
            self.model = ViTs_face(
                loss_type='CosFace',
                GPU_ID=device,
                num_class=93431,
                image_size=112,
                patch_size=8,
                ac_patch_size=12,
                pad=4,
                dim=512,
                depth=20,
                heads=8,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1)
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        self.model = self.model.to(device).eval()

        self.preprocessing = transforms.Compose([
            transforms.Resize((112, 112))
        ])
        self.img_size = 112
        self.drange = 255
        self.dis_func = "cosine"
        self.embedding_dim = 512
        self.l2norm = True

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        flipped_imgs = torch.flip(imgs, dims=[3])
        return F.normalize(self.model(imgs) + self.model(flipped_imgs))
    
class FaceViT(object):
    """
    Model: FaceViT (Hybrid ViT with ResNet blocks. For certain models requiring two inputs, cross-image attention is used.)
    Dataset: WebFace2M
    Loss Function: ArcFace
    Source: https://github.com/anguyen8/face-vit (WACV 2023)

    Args:
        cross_image_attention (bool): Whether to use cross-image attention. If True, the model requires even-numbered batch size. 
        depth (int): Depth of the Transformer.
        heads (int): Number of attention heads.
        device (torch.device): The device to run the model on.

    Note:
        The supported combinations of (cross_image_attention, depth, heads) are:
        - (False, 1, 1), (False, 1, 2), (False, 1, 4), (False, 1, 6), (False, 1, 8)
        - (False, 2, 1), (False, 2, 2), (False, 2, 4), (False, 2, 6), (False, 2, 8)
        - (False, 4, 1), (False, 4, 2), (False, 4, 4), (False, 4, 6), (False, 4, 8)
        - (False, 8, 1), (False, 8, 2)
        - (True, 1, 6), (True, 1, 8)
        - (True, 2, 1), (True, 2, 2), (True, 2, 4), (True, 2, 6), (True, 2, 8)
        - (True, 4, 1), (True, 4, 2), (True, 4, 4), (True, 4, 6), (True, 4, 8)
        - (True, 6, 1), (True, 6, 2), (True, 6, 4), (True, 6, 6), (True, 6, 8)
        - (True, 8, 1), (True, 8, 2)
        Currently, only the weight urls of (False, 8, 1), (False, 8, 2), (True, 8, 1), (True, 8, 2) are provided for download. 
        (False, 8, 1) and (True, 8, 1) are provided as examples in the test.py of the original repository.
    """
    def __init__(self, cross_image_attention: bool, depth: int, heads: int, device: torch.device):
        # single Image input with Hybrid ViT
        self.device = device
        self.cross_image_attention = cross_image_attention
        #self.gray_scale = True if not self.cross_image_attention else False
        self.path_dict = {}
        for d in [1, 2, 4]:
            for h in [1, 2, 4, 6, 8]:
                self.path_dict[f'single_d{d}h{h}'] = os.path.join(FACEVIT_HOME, f'pretrained/ViT-P8S8_2-image_webface_2m_arcface_resnet18_s1_depth_{d}_head_{h}_hybrid/best.pth')
        self.path_dict['single_d8h1'] = os.path.join(FACEVIT_HOME, 'pretrained/ViT-P8S8_2-image_webface_2m_arcface_resnet18_s1_depth_8_head_1_hybrid/best.pth')
        self.path_dict['single_d8h2'] = os.path.join(FACEVIT_HOME, 'pretrained/ViT-P8S8_2-image_webface_2m_arcface_resnet18_s1_depth_8_head_2_hybrid/best.pth')
        for d in [2, 4, 6]:
            for h in [1, 2, 4, 6, 8]:
                self.path_dict[f'cross_d{d}h{h}'] = os.path.join(FACEVIT_HOME, f'pretrained/ViT-P8S8_2-image_webface_2m_arcface_resnet18_s1_depth_{d}_head_{h}_LFW_lr1e5/best.pth')
        self.path_dict['cross_d1h6'] = os.path.join(FACEVIT_HOME, 'pretrained/ViT-P8S8_2-image_webface_2m_arcface_resnet18_s1_depth_1_head_6_LFW_lr1e5/best.pth')
        self.path_dict['cross_d1h8'] = os.path.join(FACEVIT_HOME, 'pretrained/ViT-P8S8_2-image_webface_2m_arcface_resnet18_s1_depth_1_head_8_LFW_lr1e5/best.pth')
        self.path_dict['cross_d8h1'] = os.path.join(FACEVIT_HOME, 'pretrained/ViT-P8S8_2-image_webface_2m_arcface_resnet18_s1_depth_8_head_1_LFW_lr1e5/best.pth')
        self.path_dict['cross_d8h2'] = os.path.join(FACEVIT_HOME, 'pretrained/ViT-P8S8_2-image_webface_2m_arcface_resnet18_s1_depth_8_head_2_LFW_lr1e5/best.pth')

        self.url_dict = {
            'single_d8h1': "https://drive.google.com/drive/folders/1Zd49TKI92hHg8mkEMzGcVTl4E2TL_WZI?usp=share_link", 
            'single_d8h2': "https://drive.google.com/drive/folders/1ce5BDYEHeQ64vDRfu8BYZLgK9FL7WwTT?usp=share_link",
            'cross_d8h1': "https://drive.google.com/drive/folders/1IEM3Tj0SLu8Zekjil4AHEIa4YqVQVyWQ?usp=share_link", 
            'cross_d8h2': "https://drive.google.com/drive/folders/1RuqEmSTzDSMj7Jq-Nk6Af_YWvqMLHH2x?usp=share_link"
        }
        model_name = f'{"cross" if self.cross_image_attention else "single"}_d{depth}h{heads}'
        problematic_models = ['single_d8h2']
        if model_name in problematic_models:
            print(f"{model_name} has (very) low baseline recall. Consider using other models.")
        if model_name in self.url_dict.keys():
            download(self.path_dict[model_name], self.url_dict[model_name])
        else:
            raise ValueError(f"Model with cross_image_attention={self.cross_image_attention}, depth={depth}, heads={heads} is currently not supported for automatic download. "
                             f"Please manually download the weights from https://drive.google.com/drive/folders/1LEshPNCEP0IGbYGXzkxNP2Tp2SUAKGzD and place them in {self.path_dict[model_name]}")
        if not self.cross_image_attention:
            self.model = Hybrid_ViT(loss_type='ArcFace',
                                    GPU_ID=device,
                                    num_class=10575,
                                    channels=1,
                                    image_size=128,
                                    patch_size=8,
                                    ac_patch_size=12,
                                    pad=4,
                                    dim=512, #256,
                                    depth=depth,
                                    heads=heads,
                                    mlp_dim=2048,
                                    dropout=0.1,
                                    emb_dropout=0.1,
                                    out_dim=512,
                                    remove_pos=False).to(device)
            model_path = self.path_dict[f'single_d{depth}h{heads}']
        else:
            self.model = CrossImage_Hybrid_ViT(loss_type='ArcFace',
                                                GPU_ID=device,
                                                num_class=10575,
                                                use_cls=False,
                                                use_face_loss=True,
                                                no_face_model=False,
                                                image_size=112,
                                                patch_size=8,
                                                ac_patch_size=12,
                                                pad=4,
                                                dim=512,
                                                depth=depth,
                                                heads=heads,
                                                mlp_dim=2048,
                                                dropout=0.1,
                                                emb_dropout=0.1,
                                                out_dim=512,
                                                singleMLP=False,
                                                remove_sep=False).to(device)
            model_path = self.path_dict[f'cross_d{depth}h{heads}']
        face_model = resnet_face18(use_se=False, use_reduce_pool=False, grayscale=True)
        self.model.face_model = face_model
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        self.model.to(device).eval()

        self.aligner = LandmarkAligner(fd_model_name='resnet50_retinaface_widerface', device=device)
        self.preprocessing = transforms.Compose([
            transforms.Lambda(lambda x: self.aligner.align_ldmks(x)),
            transforms.Resize((128, 128)), 
            transforms.Grayscale(num_output_channels=1),
            transforms.Lambda(lambda x: (x - 127.5) / 127.5)
        ])
        self.img_size = 128 
        self.drange = 255
        self.dis_func = "cosine"
        self.embedding_dim = 512
        self.l2norm = True

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        if not self.cross_image_attention:
            embeds = self.model(imgs)
            flipped_embeds = self.model(torch.flip(imgs, dims=[3]))
            return F.normalize(embeds + flipped_embeds, p=2, dim=1)
        else:
            return F.normalize(self.model(torch.cat([imgs, torch.flip(imgs, [0])], dim=0), fea=True)[0], p=2, dim=1)
            """embeds = []
            for idx in range(0, imgs.shape[0], 2):
                img1 = imgs[idx]
                out_of_bound = idx + 1 >= imgs.shape[0]
                if out_of_bound:
                    img2 = imgs[0]
                else:
                    img2 = imgs[idx + 1]
                embed1, embed2 = self.model(torch.stack([img1, img2], dim=0), fea=True)
                embeds.extend([embed1[0], embed2[0]]) if not out_of_bound else embeds.append(embed1[0])
            return F.normalize(torch.stack(embeds, dim=0), p=2, dim=1)"""
                
class LandmarkAligner(object):
    def __init__(self, fd_model_name: str, device: torch.device):
        self.device = device
        self.fd = FD(model_name=fd_model_name, device=device)
        self.ldmk_template_np = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        self.ldmk_template = torch.tensor(self.ldmk_template_np, dtype=torch.float32, device=device).to(device)  # (5, 2)
        self.ldmk_template[:, 0] += 8.0
        self.ldmk_template_np[:, 0] += 8.0
    
    def align_ldmks(self, img: torch.Tensor) -> torch.Tensor:
        """
        imgs: (3, H, W), float32, [0, 255] or [0, 1]
        """
        dets = self.fd(img.clone().detach())
        if dets is None or len(dets) == 0:
            return img
        largest_face, ldmks = -1, None
        for det in dets:
            bbox = det[:4]
            score = det[4]
            cur_ldmks: List[int] = det[5]
            size = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]) * score
            if size > largest_face:
                largest_face = size
                ldmks = cur_ldmks
        if ldmks is None:
            return img
        if len(ldmks) == 10:
            ldmks: np.ndarray = np.array(ldmks, dtype=np.float32).reshape(5, 2)
        elif len(ldmks) == 136:
            ldmks = np.array(ldmks, dtype=np.float32).reshape(68, 2)
            ldmks = self.ldmk68_to_5(ldmks)
        else:
            raise ValueError(f"Unsupported landmark number: {len(ldmks) // 2}")
        tform_estimator = skitrans.SimilarityTransform()
        success = tform_estimator.estimate(ldmks, self.ldmk_template_np)
        if not success:
            ldmks: torch.Tensor = torch.tensor(ldmks, dtype=torch.float32, device=self.device).to(self.device)  # (5, 2)
            tform = self.estimate_sim_matrix(ldmks.unsqueeze(0), self.ldmk_template.unsqueeze(0))  # (1, 2, 3)
        else:
            tform = torch.tensor(tform_estimator.params[0:2, :], dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 2, 3)
        aligned = kornia.geometry.transform.warp_affine(img.unsqueeze(0), tform, dsize=(112, 112), mode='bilinear', padding_mode='zeros', align_corners=True)
        ####################### Visualization for debugging #######################
        """print(aligned.shape, img.shape)
        print(img.max(), img.min(), img.mean())
        print(aligned.max(), aligned.min(), aligned.mean())
        _aligned = cv2.cvtColor(aligned.clone().detach().mul(255.).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.putText(_aligned, "Aligned", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        _img = cv2.cvtColor(img.clone().detach().mul(255.).permute(1, 2, 0).cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.putText(_img, "Original", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if not success:
            ldmks = ldmks.clone().detach().cpu().numpy()
        ldmks = ldmks.astype(np.int32)
        for (x, y) in ldmks:
            cv2.circle(_img, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in self.ldmk_template_np:
            cv2.circle(_aligned, (int(x), int(y)), 2, (0, 255, 0), -1)
        print(_img.shape, _aligned.shape)
        print(np.max(_img), np.min(_img), np.mean(_img))
        print(np.max(_aligned), np.min(_aligned), np.mean(_aligned))
        _img = cv2.resize(_img, (112, 112))
        _frame = np.hstack([_img, _aligned])
        cv2.imwrite(f"/home/zlwang/ProtegoPlus/trash/aligned/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png", _frame)"""
        #############################################################################
        return aligned.squeeze(0)

    @staticmethod
    def ldmk68_to_5(ldmk68: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(ldmk68, np.ndarray):
            lm5 = np.zeros((5, 2), dtype=np.float32)
        elif isinstance(ldmk68, torch.Tensor):
            lm5 = torch.zeros(5, 2, device=ldmk68.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported landmark type in PartViT: {type(ldmk68)}")
        lm5[0] = (ldmk68[36] + ldmk68[39]) / 2.0   
        lm5[1] = (ldmk68[42] + ldmk68[45]) / 2.0   
        lm5[2] = ldmk68[30]                           
        lm5[3] = ldmk68[48]                            
        lm5[4] = ldmk68[54]                            
        return lm5

    @staticmethod
    def estimate_sim_matrix(src_pts: torch.Tensor, dst_pts: torch.Tensor) -> torch.Tensor:
        # 1. Demean
        src_mean = src_pts.mean(dim=1, keepdim=True)   # (B,1,2)
        dst_mean = dst_pts.mean(dim=1, keepdim=True)
        src_centered = src_pts - src_mean
        dst_centered = dst_pts - dst_mean
        # 2. Calculate variance for scale
        var_src = (src_centered ** 2).sum(dim=[1,2])  # (B,)
        # Covariance matrix H = src_centered^T * dst_centered
        H = torch.matmul(src_centered.transpose(1,2), dst_centered)  # (B,2,2)
        # 3. SVD to get rotation
        U, S, Vt = torch.linalg.svd(H)
        R = torch.matmul(Vt.transpose(-1,-2), U.transpose(-1,-2))  # (B,2,2)
        # Deal with reflection case
        det = torch.linalg.det(R)
        mask = det < 0
        if mask.any():
            Vt_adj = Vt.clone()
            Vt_adj[mask, -1, :] *= -1
            R = torch.matmul(Vt_adj.transpose(-1,-2), U.transpose(-1,-2))
        # 4. Calculate scale
        scale = (S.sum(dim=1) / var_src).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
        # 5. Calculate translation
        t = dst_mean.transpose(1,2) - scale * torch.matmul(R, src_mean.transpose(1,2))  # (B,2,1)
        # 6. Compose transformation matrix
        return torch.cat([scale * R, t], dim=2)  # (B,2,3)

class PartfViT(object):
    """
    Model: Part-based fViT (Hybrid ViT with patches based on the face landmarks given by MobileNetV3)
    Datasets: WebFace4M or 1-shot WebFace4M
    Loss Function: CosFace
    Other:
        'nosl' means no pretraining
        'lafssl' means pretrained with LAFS and retrained with Supervised Learning
        'dinossl' means self-supervised learning with DINO
        'lafsssl' means self-supervised learning with LAFS
    Source: https://github.com/szlbiubiubiu/LAFS_CVPR2024 (CVPR2024)
    Special Warning: 
        This model requires face alignment using 5 facial landmarks as preprocessing and many operations are not differentiable.
        Therefore, it is recommended to use this model only for inference and not for training. 
        But in theory, if the gradient only flows from the input images to the output features, it should be fine. This is not tested yet.
        BTW, it is tested that the model can work well even without face alignment. You may choose to disable it in the code manually.

    Args:
        model_name (str): Model name. Options are 'partfvit_cosface_nosl_webface', 'partfvit_cosface_lafssl_webface', 'fvit_cosface_dinossl_webface', 'partfvit_cosface_lafsssl_webface'.
        device (torch.device): The device to run the model on.
    """
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.path_dict = {
            'partfvit_cosface_nosl_webface': os.path.join(PARTFVIT_HOME, 'pretrained/webface_196land_sp.pth'),
            'partfvit_cosface_nosl_ms1mv3': os.path.join(PARTFVIT_HOME, 'pretrained/part_vit_B_34epoch.pth'),
            'partfvit_cosface_lafssl_webface': os.path.join(PARTFVIT_HOME, 'pretrained/lafs_webface_finetune_withaugmentation.pth'),
            'fvit_cosface_dinossl_webface': os.path.join(PARTFVIT_HOME, 'pretrained/SSL_Webface_ViTB.pth'),
            'partfvit_cosface_lafsssl_webface': os.path.join(PARTFVIT_HOME, 'pretrained/SSL_Webface_webland_partViTB.pth')
        }
        self.url_dict = {
            'partfvit_cosface_nosl_webface': "https://drive.google.com/file/d/16fsYE-j4v6dh7V-_aM0nnU9VdjjlZ1VX/view?usp=share_link",
            'partfvit_cosface_nosl_ms1mv3': "https://drive.google.com/file/d/1ev-y0aOmt1mhQCCZwh3ef204ibszi1Rl/view?usp=share_link", 
            'partfvit_cosface_lafssl_webface': "https://drive.google.com/file/d/1BUYm2Bcgp8ZRlBcwOZxiJtWiQAvK2Ujy/view?usp=share_link", 
            'fvit_cosface_dinossl_webface': "https://drive.google.com/file/d/19hbQYNnMvJ5enKlxOQnb5QSCefL6MTuA/view?usp=share_link", 
            'partfvit_cosface_lafsssl_webface': "https://drive.google.com/file/d/1WykUT8MRBbc8Oc-WjQ_aya2ubfLPMgae/view?usp=share_link"
        }
        download(self.path_dict[model_name], self.url_dict[model_name])
        model_path = self.path_dict[model_name]
        if model_name == 'partfvit_cosface_nosl_webface':
            self.model = ViT_face_landmark_patch8(
                            loss_type='CosFace',
                            GPU_ID=device,
                            num_class=205990,
                            image_size=112,
                            patch_size=8,#8 14
                            dim=768,#512 ,768
                            depth=12,#20,12
                            heads=11,
                            mlp_dim=2048,
                            dropout=0.1,
                            emb_dropout=0.1,
                            with_land=True).to(device).eval()
        elif model_name == 'partfvit_cosface_nosl_ms1mv3':
            self.model = ViT_face_landmark_patch8(
                        loss_type='CosFace',
                        GPU_ID=None,
                        num_class=93431,
                        num_patches=196, 
                        image_size=112,
                        patch_size=8, 
                        dim=768,
                        depth=12,
                        heads=11,
                        mlp_dim=2048,
                        dropout=0.1,
                        emb_dropout=0.1,
                        with_land=True).to(device).eval()
        else:
            raise NotImplementedError(f"Have not found the corresponding model for {model_name}, despite the availability of the pretrained weights. This will be fixed in future versions.")
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False), strict=True)

        self.aligner = LandmarkAligner(fd_model_name='resnet50_retinaface_widerface', device=device)
        self.preprocessing = transforms.Compose([
            transforms.Lambda(lambda x: self.aligner.align_ldmks(x)),
            transforms.Resize((112, 112)),
            transforms.Lambda(lambda x: (x - 0.5)) # ! Range [-0.5, 0.5]
        ])
        # According to my test, partfvit_cosface_nosl_webface favors no alignment and [-1, 1] a little more. 
        # But the difference is very small. 
        # Therefore, I decided to keep keep the process shown in the IJB_evaluation.py of the original repo. 
        self.img_size = 112
        self.drange = 1
        self.dis_func = "cosine"
        self.embedding_dim = 768
        self.l2norm = True
    
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        flipped = torch.flip(imgs, dims=[3])
        feat_orig = self.model(imgs)
        feat_flip = self.model(flipped)
        feats = feat_orig + feat_flip
        return F.normalize(feats, p=2, dim=1)

class TransFace(object):
    """
    Model: TransFace (Pure ViT with patchify done by convolution and weighing patches with SENet-style module)
    Datasets: MS1MV2 / Glint360k
    Loss Function: ArcFace
    Source: https://github.com/DanJun6737/TransFace (ICCV2023)
    Args:
        size (str): Model size. Options are 's', 'b', 'l'.
        pretrained (str): Pretrained model name. Options are 'ms1mv2', 'glint360k'.
        device (torch.device): The device to run the model on.
    """
    def __init__(self, size: str, pretrained: str, device: torch.device):
        self.path_dict = {
            's_ms1mv2': os.path.join(TRANSFACE_HOME, 'pretrained/ms1mv2_model_TransFace_S.pt'),
            'b_ms1mv2': os.path.join(TRANSFACE_HOME, 'pretrained/ms1mv2_model_TransFace_B.pt'),
            'l_ms1mv2': os.path.join(TRANSFACE_HOME, 'pretrained/ms1mv2_model_TransFace_L.pt'),
            's_glint360k': os.path.join(TRANSFACE_HOME, 'pretrained/glint360k_model_TransFace_S.pt'),
            'b_glint360k': os.path.join(TRANSFACE_HOME, 'pretrained/glint360k_model_TransFace_B.pt'),
            'l_glint360k': os.path.join(TRANSFACE_HOME, 'pretrained/glint360k_model_TransFace_L.pt')
        }
        self.url_dict = {
            's_ms1mv2': "https://drive.google.com/file/d/1UZWCg7jNESDv8EWs7mxQSswCMGbAZNF4/view?usp=share_link",
            'b_ms1mv2': "https://drive.google.com/file/d/16O-q30mH8d3lECqa5eJd8rABaUlNhQ0K/view?usp=share_link",
            'l_ms1mv2': "https://drive.google.com/file/d/1uXUFT6ujEPqvCTHzONsp6-DMIc24Cc85/view?usp=share_link",
            's_glint360k': "https://drive.google.com/file/d/18Zh_zMlYttKVIGArmDYNEchIvUSH5FQ1/view?usp=share_link",
            'b_glint360k': "https://drive.google.com/file/d/13IezvOo5GvtGVsRap2s5RVqtIl1y0ke5/view?usp=share_link",
            'l_glint360k': "https://drive.google.com/file/d/1jXL_tidh9KqAS6MgeinIk2UNWmEaxfb0/view?usp=share_link"
        }
        model_name = f"{size}_{pretrained}"
        download(self.path_dict[model_name], self.url_dict[model_name])
        model_path = self.path_dict[model_name]
        self.model = get_transface_model(name=f"vit_{size}_dp005_mask_005", dropout=0, fp16=False).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        self.model.eval()

        self.aligner = LandmarkAligner(fd_model_name='resnet50_retinaface_widerface', device=device)
        self.preprocessing = transforms.Compose([
            transforms.Lambda(lambda x: self.aligner.align_ldmks(x)),
            transforms.Resize((112, 112)),
            transforms.Lambda(lambda x: (x - 0.5) / 0.5)
        ])
        self.img_size = 112
        self.drange = 1
        self.dis_func = "cosine"
        self.embedding_dim = 512
        self.l2norm = True

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        flipped = torch.flip(imgs, dims=[3])
        feat_orig = self.model(imgs)[0]
        feat_flip = self.model(flipped)[0]
        feats = feat_orig + feat_flip
        return F.normalize(feats, p=2, dim=1)

class SwinFace(object):
    """
    Model: SwinFace (Pure ViT but with CNN-style hierarchical architecture and patch merging)
    Datasets: MS1MV2
    Loss Function: ArcFace
    Source: https://github.com/lxq1000/SwinFace (TCSVT2024, CCF-B)
    Args:
        size (str): Model size. Options are 's', 'b', 'l'.
        pretrained (str): Pretrained model name. Options are 'ms1mv2', 'glint360k'.
        device (torch.device): The device to run the model on.
    """
    def __init__(self, model_name: str, device: torch.device):
        self.path_dict = {
            'swin_t': os.path.join(SWINFACE_HOME, 'pretrained/checkpoint_step_79999_gpu_0.pt')
        }
        self.url_dict = {
            'swin_t': "https://drive.google.com/file/d/1fi4IuuFV8NjnWm-CufdrhMKrkjxhSmjx/view?usp=share_link"
        }
        download(self.path_dict[model_name], self.url_dict[model_name])
        model_path = self.path_dict[model_name]
        self.model = build_swinface_model(model_name=model_name).to(device)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        self.model.backbone.load_state_dict(ckpt["state_dict_backbone"])
        self.model.fam.load_state_dict(ckpt["state_dict_fam"])
        self.model.tss.load_state_dict(ckpt["state_dict_tss"])
        self.model.om.load_state_dict(ckpt["state_dict_om"])
        self.model.eval()

        self.preprocessing = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.Lambda(lambda x: (x - 0.5) / 0.5) # ! Range [-1, 1]
        ])
        self.img_size = 112
        self.drange = 1
        self.dis_func = "cosine"
        self.embedding_dim = 512
        self.l2norm = True

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model(imgs)["Recognition"], p=2, dim=1)

"""
import os

import torch
import torch.nn.functional as F
import cv2

from protego.FacialRecognition import FR
from protego.utils import load_imgs

with torch.no_grad():
    device = "cuda:0"
    fr = FR(model_name="facevit_arcface_singled8h1_webface", device=device)
    base_dir = "/home/zlwang/ProtegoPlus/face_db/face_scrub/Bradley_Cooper"
    imgs = [os.path.join(base_dir, n) for n in os.listdir(base_dir) if n.lower().endswith(('.png', '.jpg', '.jpeg')) and not n.startswith(('.', '_'))]
    imgs = load_imgs(img_paths=imgs, device=device, img_sz=224)
    img_nums = len(imgs)
    embs = fr(imgs)
    print(embs.shape, embs.norm(dim=1))
    similarity_matrix = F.normalize(embs) @ F.normalize(embs).T - torch.eye(img_nums, device=device)
    print(similarity_matrix.sum() / (len(embs) ** 2 - len(embs)))
    print(similarity_matrix.max(), similarity_matrix.min())
"""