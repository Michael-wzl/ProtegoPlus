import os
import gc
import sys
from typing import Tuple, List
import datetime
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" 

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
import torch.nn.functional as F
import cv2
from PIL import Image
from skimage.transform import warp
import numpy as np
import tqdm
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker

from smirk.src.smirk_encoder import SmirkEncoder
from smirk.src.FLAME.FLAME import FLAME
from .UVGenerator import Renderer
from .utils import img2tensor, BASE_PATH
from .setup_user_uvs import run_mediapipe, crop_face, gen_uvs, restore, init_flame, init_renderer, init_smirk, init_mp_lmk_detector
from mtcnn_pytorch.src.detector import detect_faces
from mtcnn_pytorch.src.get_nets import PNet, RNet, ONet

def coarse_crop(img: Image, 
                pnet: PNet, 
                rnet: RNet,
                onet: ONet, 
                conf_thresh: float = 0.8, 
                min_h: int = 20, 
                min_w: int = 20) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop the face from the image using MTCNN.

    Args:
        img (Image): The input image.
        pnet (PNet): The PNet model for face detection.
        rnet (RNet): The RNet model for face detection.
        onet (ONet): The ONet model for face detection.
        conf_thresh (float): Confidence threshold for face detection.
        min_h (int): Minimum height of the detected face.
        min_w (int): Minimum width of the detected face.

    Returns:
        Tuple[np.ndarray, Tuple[int, int, int, int]]: The cropped face and its position in the original image.
    """
    bboxs, _ = detect_faces(img, pnet=pnet, rnet=rnet, onet=onet)
    if len(bboxs) == 0:
        return None, None
    img_np = np.array(img)
    height, width = img_np.shape[:2]
    cropped_face = None
    position = None
    largest_area = 0
    for bbox in bboxs:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        conf = bbox[4] 
        if conf < conf_thresh:  # Confidence threshold
            print(f"Discarding face with low confidence: {conf:.2f}")
            continue
        face_width = x2 - x1
        face_height = y2 - y1
        if face_height < min_h or face_width < min_w:
            print(f"Discarding face with too small dimensions: width={face_width}, height={face_height}")
            continue
        face_area = face_width * face_height
        if face_area < largest_area:
            continue
        largest_area = face_area

        square_size = max(face_width, face_height)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        half_size = square_size // 2

        square_x1 = center_x - half_size
        square_y1 = center_y - half_size
        square_x2 = center_x + half_size
        square_y2 = center_y + half_size

        pad_left = max(0, -square_x1)
        pad_top = max(0, -square_y1)
        pad_right = max(0, square_x2 - width)
        pad_bottom = max(0, square_y2 - height)

        src_x1 = max(0, square_x1)
        src_y1 = max(0, square_y1)
        src_x2 = min(width, square_x2)
        src_y2 = min(height, square_y2)

        if len(img_np.shape) == 3:
            square_face = np.full((square_size, square_size, 3), 128, dtype=img_np.dtype)
        else:
            square_face = np.full((square_size, square_size), 128, dtype=img_np.dtype)

        dst_x1 = pad_left
        dst_y1 = pad_top
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        if len(img_np.shape) == 3:
            square_face[dst_y1:dst_y2, dst_x1:dst_x2, :] = img_np[src_y1:src_y2, src_x1:src_x2, :]
        else:
            square_face[dst_y1:dst_y2, dst_x1:dst_x2] = img_np[src_y1:src_y2, src_x1:src_x2]

        cropped_face = square_face
        position = (square_x1, square_y1, square_x2, square_y2)

    return cropped_face, position

def get_uvs(img_np: np.ndarray, img_sz: int, lmk_detector: FaceLandmarker, smirk_encoder: SmirkEncoder, flame: FLAME, renderer: Renderer, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate UVs and binary mask from the cropped image.

    Args:
        img_np (np.ndarray): The cropped image as a numpy array.
        img_sz (int): The size to which the image will be resized.
        lmk_detector (FaceLandmarker): The MediaPipe face landmark detector.
        smirk_encoder (SmirkEncoder): The SMIRK encoder for generating UVs.
        flame (FLAME): The FLAME model for face modeling.
        renderer (Renderer): The renderer for generating the final UVs and masks.
        device (torch.device): The device to run the computations on.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The generated UVs and binary mask.
    """
    kpt_mp = run_mediapipe(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), lmk_detector)
    if kpt_mp is None:
        return None, None
    kpt_mp = kpt_mp[..., :2]
    tform = crop_face(img_np,kpt_mp,scale=1.4,image_size=img_sz)
    cropped_img_np = warp(img_np, tform.inverse, output_shape=(img_sz, img_sz), preserve_range=True).astype(np.uint8)
    cropped_img_pt = img2tensor(cropped_img_np).to(device)
    uv, bin_mask = gen_uvs(cropped_img_pt, smirk_encoder, flame, renderer, device)
    uv = torch.tensor(restore(uv.cpu().numpy(), np.expand_dims(tform.params, axis=0), img_sz), dtype=torch.float32, device=device)
    bin_mask = torch.tensor(restore(bin_mask.cpu().numpy(), np.expand_dims(tform.params, axis=0), img_sz), dtype=torch.float32, device=device)
    bin_mask = bin_mask.permute(0, 3, 1, 2)  # [1, 1, 224, 224]
    return uv, bin_mask

def gen_vids(mask: torch.Tensor, 
            epsilon: float, 
            use_bin_mask: bool, 
            compare_mode: bool, 
            scr_vid_path: str, 
            dst_vid_path: str, 
            device: torch.device, 
            pnet: PNet,
            rnet: RNet, 
            onet: ONet,
            img_sz: int, 
            lmk_detector: FaceLandmarker, 
            smirk_encoder: SmirkEncoder, 
            flame: FLAME, 
            renderer: Renderer) -> None:
    """
    Generate a video with protected frames using the given mask and MTCNN for face detection.

    Args:
        mask (torch.Tensor): The perturbation mask tensor. Shape should be [1, 3, H, W].
        epsilon (float): The perturbation strength.
        compare_mode (bool): Whether to put the unprotected and protected frames side by side.
        scr_vid_path (str): Path to the source video.
        dst_vid_path (str): Path to save the protected video.
        device (torch.device): The device to run the computations on.
        pnet (PNet): The PNet model for face detection.
        rnet (RNet): The RNet model for face detection.
        onet (ONet): The ONet model for face detection.
        img_sz (int): The size to which the image will be resized.
        lmk_detector (FaceLandmarker): The MediaPipe face landmark detector.
        smirk_encoder (SmirkEncoder): The SMIRK encoder for generating UVs.
        flame (FLAME): The FLAME model for face modeling.
        renderer (Renderer): The renderer for generating the final UVs and masks.
    """
    # Init the readers and writers
    scr_vid = cv2.VideoCapture(scr_vid_path)
    f_rate = scr_vid.get(cv2.CAP_PROP_FPS)
    H, W = int(scr_vid.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(scr_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    dst_sz = (W, H) if not compare_mode else (W*3, H)
    dst_vid = cv2.VideoWriter(dst_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), f_rate, dst_sz)
    # Prepare the recorders
    frame_cnt = 0
    total_time = datetime.timedelta(0)

    for frame_idx in tqdm.tqdm(range(int(scr_vid.get(cv2.CAP_PROP_FRAME_COUNT))), desc="Generating video"):
        ret, frame = scr_vid.read()
        
        if not ret:
            break
        # Crop the image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame)
        face, position = coarse_crop(pil_frame, pnet, rnet, onet)
        if face is None or position is None:
            print(f"No face detected in frame {frame_idx}. Skipping...")
            continue
        orig_h, orig_w = face.shape[:2]
        if orig_h == 0 or orig_w == 0:
            print(f"Face crop has zero size in frame {frame_idx}. Skipping...")
            continue
        if (position[1] < 0 or position[0] < 0 or
            position[1] + orig_h > H or position[0] + orig_w > W):
            print(f"Face crop out of bounds in frame {frame_idx}. Skipping...")
            continue
        start_time = datetime.datetime.now()
        img_np = cv2.resize(face, (img_sz, img_sz), interpolation=cv2.INTER_LANCZOS4 if orig_h > img_sz else cv2.INTER_AREA)
        uv, bin_mask = get_uvs(img_np, img_sz, lmk_detector, smirk_encoder, flame, renderer, device)
        if uv is None or bin_mask is None:
            print(f"No UV or Segmentation Mask can be generated with SMIRK in frame {frame_idx}. Skipping...")
            continue
        # Generate the perturbations
        perturbation = torch.clamp(F.grid_sample(mask.clone(), uv, mode='bilinear', align_corners=True), -epsilon, epsilon)
        if use_bin_mask:
            perturbation *= bin_mask
        perturbation = torch.clamp(F.interpolate(perturbation, size=(orig_h, orig_w), mode='bilinear', align_corners=True), -epsilon, epsilon).squeeze(0)
        # Apply the perturbations and create the protected frames
        protected_frame_pt = img2tensor(frame.copy()).to(device).squeeze(0) 
        face_area = protected_frame_pt[:, position[1] : position[1] + orig_h, position[0] : position[0] + orig_w].clone()
        face_area = torch.clamp(face_area + perturbation, 0, 1)
        protected_frame_pt[:, position[1] : position[1] + orig_h, position[0] : position[0] + orig_w] = face_area
        protected_frame = protected_frame_pt.mul(255.).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # Create the final perturbation frames
        if compare_mode:
            perturbation = ((perturbation - perturbation.min()) / (perturbation.max() - perturbation.min())).mul(255.).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            perturbation_frame = np.full_like(protected_frame, 128)  # shape: [H, W, 3]
            perturbation_frame[position[1]:orig_h + position[1], position[0]:orig_w + position[0], :] = perturbation
            
        end_time = datetime.datetime.now()
        frame_cnt += 1
        total_time += (end_time - start_time)

        if compare_mode:
            final_frame = np.zeros((H, W * 3, 3), dtype=np.uint8)
            final_frame[:, :W, :] = frame
            final_frame[:, W:2*W, :] = protected_frame
            final_frame[:, 2*W:3*W, :] = perturbation_frame
            final_frame = cv2.putText(final_frame, f"Original Frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            final_frame = cv2.putText(final_frame, f"Protected Frame ({scr_vid_path})", (W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            final_frame = cv2.putText(final_frame, f"Perturbation ({scr_vid_path})", (2 * W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            final_frame = protected_frame
        final_frame = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
        dst_vid.write(final_frame)
    scr_vid.release()
    dst_vid.release()

    print(f"Processed {frame_cnt} frames in {total_time}. Inference FPS: {frame_cnt / total_time.total_seconds():.2f}, Original Video FPS: {f_rate:.2f}")

if __name__ == "__main__":
    with torch.no_grad():
        ######################### Configuration #########################
        device = torch.device("cuda:0") 
        protectee = "Bradley_Cooper"  # The name of the protectee
        vid_name = "Bradley_Cooper"  # The name of the video to be protected
        mask_exp_name = ["default", "frpair0_mask0_univ_mask"] # The name of the experiment that contains the mask to be used
        epsilon = 16 # ! This has to correspond to the epsilon used in the mask generation, refer the saved config file
        use_bin_mask = True # ! This has to correspond to the epsilon used in the mask generation, refer the saved config file
        compare_mode = True # Whether to put the unprotected and protected frames side by side. If False, only protected frames will be saved.
        #################################################################
        scr_vid_path = f"{BASE_PATH}/face_db/vids/{protectee}/{vid_name}.mp4"
        os.makedirs(os.path.join(BASE_PATH, 'results', 'vids', protectee), exist_ok=True)
        dst_vid_path = f"{BASE_PATH}/results/vids/{protectee}/{vid_name}_protected.mp4" if not compare_mode else f"{BASE_PATH}/results/vids/{protectee}/{vid_name}_compare.mp4"
        mask_src_path = f"{BASE_PATH}/experiments/{mask_exp_name[0]}/{protectee}/{mask_exp_name[1]}.npy"
        img_sz = 224
        epsilon /= 255. 
        mask = torch.tensor(np.load(mask_src_path), dtype=torch.float32, device=device).to(device)[[0]]
        pnet = PNet().eval()
        rnet = RNet().eval()
        onet = ONet().eval()
        smirk_base_path = os.path.join(BASE_PATH, 'smirk')
        smirk_weight_path = os.path.join(smirk_base_path, 'pretrained_models/SMIRK_em1.pt')
        mp_lmk_model_path = os.path.join(smirk_base_path, 'assets/face_landmarker.task')
        smirk_encoder = init_smirk(smirk_weight_path, device)
        flame = init_flame(smirk_base_path, device)
        renderer = init_renderer(smirk_base_path, device)
        lmk_detector = init_mp_lmk_detector(mp_lmk_model_path)

        gen_vids(mask=mask, 
                epsilon=epsilon,
                use_bin_mask=use_bin_mask,
                compare_mode=compare_mode,
                scr_vid_path=scr_vid_path,
                dst_vid_path=dst_vid_path,
                device=device,
                pnet=pnet,
                rnet=rnet, 
                onet=onet,
                img_sz=img_sz,
                lmk_detector=lmk_detector,
                smirk_encoder=smirk_encoder,
                flame=flame,
                renderer=renderer)