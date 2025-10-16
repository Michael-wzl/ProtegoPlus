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