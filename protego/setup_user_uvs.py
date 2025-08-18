import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" 
from typing import Tuple, List
import datetime

import torch
from torch.utils.data import DataLoader
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
import cv2
from PIL import Image
from skimage.transform import estimate_transform, warp
import numpy as np
import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker

from smirk.src.smirk_encoder import SmirkEncoder
from smirk.src.FLAME.FLAME import FLAME
from .UVGenerator import Renderer
from .utils import img2tensor, complete_del, BASE_PATH

def init_smirk(smirk_ckpts_path: str, device: torch.device) -> SmirkEncoder:
    """
    Initialize the SMIRK encoder with the provided checkpoint path and device.

    Args:
        smirk_ckpts_path (str): Path to the SMIRK checkpoint file.
        device (torch.device): The device to load the model onto (e.g., 'cuda' or 'cpu').
    """
    smirk_encoder = SmirkEncoder().to(device)
    ckpts = torch.load(smirk_ckpts_path, map_location=device, weights_only=False)
    checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in ckpts.items() if 'smirk_encoder' in k} # checkpoint includes both smirk_encoder and smirk_generator
    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()
    return smirk_encoder

def init_flame(smirk_base_path: str, device: torch.device) -> FLAME:
    return FLAME(base_path=smirk_base_path).to(device)

def init_renderer(smirk_base_path: str, device: torch.device) -> Renderer:
    return Renderer(smirk_path=smirk_base_path).to(device)

def init_mp_lmk_detector(model_asset_path: str = 'assets/face_landmarker.task') -> FaceLandmarker:
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1,
                                        min_face_detection_confidence=0.1,
                                        min_face_presence_confidence=0.1
                                        )
    detector = vision.FaceLandmarker.create_from_options(options)
    return detector

def run_mediapipe(image: np.ndarray, detector: FaceLandmarker) -> np.ndarray:
    # print(image.shape)    
    image_numpy = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    # STEP 3: Load the input image.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_numpy)


    # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect(image)

    if len(detection_result.face_landmarks) == 0:
        print('No face detected')
        return None
    
    face_landmarks = detection_result.face_landmarks[0]

    face_landmarks_numpy = np.zeros((478, 3))

    for i, landmark in enumerate(face_landmarks):
        face_landmarks_numpy[i] = [landmark.x*image.width, landmark.y*image.height, landmark.z]

    return face_landmarks_numpy

def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform

def gen_uvs(imgs: torch.Tensor, smirk_encoder: SmirkEncoder, flame: FLAME, renderer: Renderer, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The function generates UV coordinates for the input images. 

    Args:
        imgs (torch.Tensor): Input images in the shape of (B, 3, 224, 224), RGB, range [0, 1], dtype torch.float32.
        smirk_encoder (SmirkEncoder): The SMIRK encoder model.
        flame (FLAME): The FLAME model.
        renderer (Renderer): The renderer model.
        device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu'). Should be the same as the device used for the models.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - uv_grid (torch.Tensor): UV coordinates in the shape of (B, H, W, 2), where H and W are the height and width of the rendered image.
            - visibility_mask (torch.Tensor): Visibility mask in the shape of (B, H, W, 1), indicating which pixels are visible.
    """
    imgs = imgs.to(device)
    smirk_outputs = smirk_encoder(imgs)
    flame_outputs = flame.forward(smirk_outputs)
    renderer_output = renderer.forward(flame_outputs['vertices'], smirk_outputs['cam'],
                                        landmarks_fan=flame_outputs['landmarks_fan'], landmarks_mp=flame_outputs['landmarks_mp'])
    uv_grid = renderer_output['uv_grid']  # [N, H, W, 2]
    visibility_mask = renderer_output['visibility_mask']  # [N, H, W, 1]
    return uv_grid, visibility_mask

def restore(tranformed: np.ndarray, tforms: np.ndarray, orig_sz: int) -> np.ndarray:
    """
    Use the transformation matrices to restore the UV maps to the original images' space.

    Args:
        tranformed (np.ndarray): The transformed UV maps in the shape of (N, H, W, C) or (N, H, W, C).
        tforms (np.ndarray): The transformation matrices in the shape of (N, 3, 3).
        orig_sz (int): The original size of the images.
    
    Returns:
        np.ndarray: The restored UV maps in the shape of (N, orig_sz, orig_sz, C).
    """
    num = tranformed.shape[0]
    restoreds = []
    for i in range(num):
        transed = tranformed[i]
        tform = tforms[i]
        restored = warp(transed, tform, output_shape=(orig_sz, orig_sz), preserve_range=True)
        restoreds.append(restored)
        #print(restored.shape, orig_sz, orig_sz)
    restoreds = np.stack(restoreds, axis=0)
    return restoreds

def setup_user_uvs(base_dir: str, lmk_detector: FaceLandmarker, smirk_encoder: SmirkEncoder, flame: FLAME, renderer: Renderer, device: torch.device, img_sz: int = 224) -> List[str]:
    """
    Generate UV maps for the images in the database and save them as tensors.

    Args:
        base_dir (str): The path to the database.
        smirk_encoder (SmirkEncoder): The SMIRK encoder model.
        flame (FLAME): The FLAME model.
        renderer (Renderer): The renderer model.
        device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        List[str]: A list of image paths for which no landmarks were detected.
    """
    no_lmks = []
    for name in sorted(os.listdir(base_dir)):
        if name.startswith('.') or name.startswith('_'):
            print(f"Skipping {name} as it starts with '.' or '_'")
            continue
        imgs = os.listdir(os.path.join(base_dir, name))
        imgs = [img for img in imgs if img.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        imgs = sorted(imgs)
        imgs_pt = []
        tforms = []
        t_start = datetime.datetime.now()
        for img_name in imgs:
            img_path = os.path.join(base_dir, name, img_name)
            with Image.open(img_path) as pil_img:
                pil_img = pil_img.convert("RGB")
                img = np.array(pil_img)
            img = cv2.resize(img, (img_sz, img_sz), interpolation=cv2.INTER_LANCZOS4 if img.shape[0] > img_sz else cv2.INTER_AREA)
            kpt_mp = run_mediapipe(img, lmk_detector)
            if kpt_mp is None:
                print(f"No landmarks detected for {img_path}. Skipping this image.")
                no_lmks.append(img_path)
                continue
            kpt_mp = kpt_mp[..., :2]
            tform = crop_face(img,kpt_mp,scale=1.4,image_size=img_sz)
            tforms.append(tform.params) # tform.params.shape = (3, 3)
            cropped_img = warp(img, tform.inverse, output_shape=(img_sz, img_sz), preserve_range=True).astype(np.uint8)
            #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4 if orig_h > 224 else cv2.INTER_AREA)
            img_tensor = img2tensor(cropped_img)
            imgs_pt.append(img_tensor)
        if len(tforms) == 0:
            print(f"No landmarks detected for any images in {name}. Skipping this user.")
            continue
        tforms = np.stack(tforms, axis=0)  # (N, 3, 3)
        imgs_pt = torch.cat(imgs_pt, dim=0)
        imgs_dl = DataLoader(imgs_pt, batch_size=64, shuffle=False)
        del imgs_pt
        complete_del()

        uvs = []
        all_visibility_masks = []
        for idx, imgs_batch in tqdm.tqdm(enumerate(imgs_dl), desc=f'Generating UVs for {name}'):
            imgs_batch = imgs_batch.to(device)
            uv_maps, visibility_masks = gen_uvs(imgs_batch, smirk_encoder, flame, renderer, device)
            uvs.append(uv_maps.cpu())
            all_visibility_masks.append(visibility_masks.cpu())
        
        uvs = torch.cat(uvs, dim=0)
        all_visibility_masks = torch.cat(all_visibility_masks, dim=0)
        # Save UV maps
        uvs_np = uvs.cpu().numpy()
        restored_uvs = torch.tensor(restore(uvs_np, tforms, img_sz), dtype=torch.float32)
        all_visibility_masks_np = all_visibility_masks.cpu().numpy()
        restored_visibility_masks = torch.tensor(restore(all_visibility_masks_np, tforms, img_sz), dtype=torch.float32)

        print(f"UVs for {name} generated in {datetime.datetime.now() - t_start}. Saving UVs and visibility masks...")
        torch.save(restored_uvs.cpu(), os.path.join(base_dir, name, 'uvs.pt'))
        torch.save(restored_visibility_masks.cpu(), os.path.join(base_dir, name, 'visibility_masks.pt'))
        
        del uvs, all_visibility_masks, restored_uvs, restored_visibility_masks, tforms, uvs_np, all_visibility_masks_np
        complete_del()

    return no_lmks

if __name__ == '__main__':
    with torch.no_grad():
        ######################### Configuration #########################
        db = os.path.join(BASE_PATH, "face_db", 'face_scrub')
        device = torch.device('cuda:0')
        #################################################################
        # Define paths for the models
        smirk_base_path = os.path.join(BASE_PATH, 'smirk')
        smirk_weight_path = os.path.join(smirk_base_path, 'pretrained_models/SMIRK_em1.pt')
        mp_lmk_model_path = os.path.join(smirk_base_path, 'assets/face_landmarker.task')


        # Init the models
        smirk_encoder = init_smirk(smirk_weight_path, device)
        flame = init_flame(smirk_base_path, device)
        renderer = init_renderer(smirk_base_path, device)
        lmk_detector = init_mp_lmk_detector(mp_lmk_model_path)

        
        # Setup the database
        no_lmk_imgs = []
        print(f"Setting up UVs for {db}...")
        no_lmk_imgs = setup_user_uvs(db, lmk_detector, smirk_encoder, flame, renderer, device)
        if no_lmk_imgs:
            print("No landmarks detected for the following images:")
            for no_lmk in no_lmk_imgs:
                print(no_lmk)
                #os.remove(no_lmk)  # Remove images with no landmarks detected
                #print(f"Removed {no_lmk} due to no landmarks detected.")
            print("Please remove these images manually from the database and rerun the script.")
        else:
            print("UVs for all images have been generated successfully.")
