import os
from typing import List, Tuple

import torch
import cv2
import numpy as np

from protego.utils import crop_face, load_imgs
from protego.FaceDetection import FD

if __name__ == "__main__":
    with torch.no_grad():
        device = torch.device('cuda:0')
        img_folder = "/home/zlwang/ProtegoPlus/face_db/BC_right"
        dst_folder = "/home/zlwang/ProtegoPlus/face_db/BC_right_cropped"
        os.makedirs(dst_folder, exist_ok=True)
        detector = FD(model_name="mtcnn", device=device)

        img_names = [os.path.join(img_folder, name) for name in os.listdir(img_folder) if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and not name.startswith(('.', '_'))]
        orig_imgs = load_imgs(img_paths=img_names, img_sz=-1, usage_portion=1., drange=255, device=device)
        for idx, img in enumerate(orig_imgs):
            img.squeeze_(0)
            face, pos = crop_face(img=img, detector=detector, verbose=True)
            if face is None or pos is None:
                print(f"Warning: No face detected in image {img_names[idx]}. Skipping...")
                continue
            face = face.permute(1, 2, 0).cpu().contiguous().numpy().astype(np.uint8)
            save_path = os.path.join(dst_folder, os.path.basename(img_names[idx]))
            print(os.path.basename(img_names[idx]), face.shape)
            cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            print(f"Cropped face saved to {save_path}.")