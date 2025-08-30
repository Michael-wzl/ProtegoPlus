import os
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
import numpy as np
import tqdm
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker

from smirk.src.smirk_encoder import SmirkEncoder
from smirk.src.FLAME.FLAME import FLAME
from mtcnn_pytorch.mtcnn import MTCNN
from .UVGenerator import Renderer
from .protect_vids import coarse_crop, get_uvs
from .setup_user_uvs import init_flame, init_renderer, init_smirk, init_mp_lmk_detector
from .utils import BASE_PATH

def protect_folder(mask: torch.Tensor, 
                epsilon: float, 
                use_bin_mask: bool, 
                scr_imgs_path: str, 
                dst_imgs_path: str, 
                device: torch.device, 
                detector: MTCNN, 
                img_sz: int, 
                lmk_detector: FaceLandmarker, 
                smirk_encoder: SmirkEncoder, 
                flame: FLAME, 
                renderer: Renderer) -> None:
    """
    Protects images in the specified folder by applying perturbations based on the provided mask.

    Args:
        mask (torch.Tensor): The perturbation mask tensor. Shape should be [1, 3, H, W].
        epsilon (float): The perturbation strength.
        use_bin_mask (bool): Whether to use the binary mask for perturbation.
        compare_mode (bool): Whether to put the unprotected and protected frames side by side.
        scr_imgs_path (str): Path to the source images.
        dst_imgs_path (str): Path to save the protected images.
        device (torch.device): The device to run the computations on.
        detector (MTCNN): The MTCNN detector.
        img_sz (int): The size to which the image will be resized.
        lmk_detector (FaceLandmarker): The MediaPipe face landmark detector.
        smirk_encoder (SmirkEncoder): The SMIRK encoder for generating UVs.
        flame (FLAME): The FLAME model for face modeling.
        renderer (Renderer): The renderer for generating the final UVs and masks.
    """
    # Init the readers and writers
    img_names = sorted([fname for fname in os.listdir(scr_imgs_path) if fname.endswith(('.jpg', '.jpeg', '.png', '.bmp')) and not fname.startswith((".", "_"))])
    frame_cnt = 0
    total_time = datetime.timedelta(0)
    for frame_idx, img_name in tqdm.tqdm(enumerate(img_names), desc="Processing images"):
        img_path = os.path.join(scr_imgs_path, img_name)
        frame = cv2.imread(img_path)
        start_time = datetime.datetime.now()
        if frame is None:
            print(f"Failed to read {img_path}. Skipping...")
            continue
        # Crop the image
        H, W, _ = frame.shape
        # Crop the image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pt = torch.tensor(frame.copy(), dtype=torch.float32, device=device).to(device).permute(2, 0, 1)
        face, position = coarse_crop(frame_pt.clone(), detector)
        face = face.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        if face is None or position is None:
            print(f"No face detected in frame {frame_idx}. Skipping...")
            continue
        orig_h, orig_w = face.shape[:2]
        if orig_h == 0 or orig_w == 0:
            print(f"Face crop has zero size in frame {frame_idx}. Skipping...")
            continue
        x0, y0 = position[0], position[1]
        if (y0 < 0 or x0 < 0 or
            y0 + orig_h > H or x0 + orig_w > W):
            print(f"Face crop out of bounds in frame {frame_idx}. Skipping...")
            continue
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
        #_perturbation = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min())
        #cv2.imwrite(f"frame_{frame_idx}_perturbation.png", cv2.cvtColor(np.ascontiguousarray(_perturbation.mul(255.).permute(1, 2, 0).cpu().numpy().astype(np.uint8)), cv2.COLOR_RGB2BGR))
        # Apply the perturbations and create the protected frames
        protected_frame_pt = frame_pt / 255.
        face_area = protected_frame_pt[:, y0 : y0 + orig_h, x0 : x0 + orig_w].clone().contiguous()
        #cv2.imwrite(f"frame_{frame_idx}_face_area_unperturbed.png", cv2.cvtColor(np.ascontiguousarray(face_area.mul(255.).permute(1, 2, 0).cpu().numpy().astype(np.uint8)), cv2.COLOR_RGB2BGR))
        face_area = torch.clamp(face_area + perturbation, 0, 1)
        #cv2.imwrite(f"frame_{frame_idx}_face_area.png", cv2.cvtColor(np.ascontiguousarray(face_area.mul(255.).permute(1, 2, 0).cpu().numpy().astype(np.uint8)), cv2.COLOR_RGB2BGR))
        protected_frame_pt[:, y0 : y0 + orig_h, x0 : x0 + orig_w] = face_area
        protected_frame = protected_frame_pt.mul(255.).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            
        end_time = datetime.datetime.now()
        frame_cnt += 1
        total_time += (end_time - start_time)
        final_frame = protected_frame
        final_frame = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(dst_imgs_path, f"{img_name.split('.')[0]}_protected.jpg"), final_frame)

    print(f"Processed {frame_cnt} images in {total_time}. Average time per image: {total_time / frame_cnt} seconds.")

if __name__ == "__main__":
    with torch.no_grad():
        ######################### Configuration #########################
        device = torch.device("cuda:0") 
        protectee = "Bradley_Cooper"  # The name of the protectee
        mask_exp_name = ["default", "frpair0_mask0_univ_mask"] # The name of the experiment that contains the mask to be used
        epsilon = 16 # ! This has to correspond to the epsilon used in the mask generation, refer the saved config file
        use_bin_mask = True # ! This has to correspond to the epsilon used in the mask generation, refer the saved config file
        #################################################################
        scr_imgs_path = f"{BASE_PATH}/face_db/imgs/{protectee}"
        dst_imgs_path = f"{BASE_PATH}/results/imgs/{protectee}"
        os.makedirs(dst_imgs_path, exist_ok=True)
        mask_src_path = f"{BASE_PATH}/experiments/{mask_exp_name[0]}/{protectee}/{mask_exp_name[1]}.npy"
        img_sz = 224
        epsilon /= 255. 
        mask = torch.tensor(np.load(mask_src_path), dtype=torch.float32, device=device).to(device)[[0]]
        detector = MTCNN(device=device, weight_paths={'pnet': os.path.join(BASE_PATH, 'mtcnn_pytorch', 'weights', 'pnet.npy'), 
                                                      'rnet': os.path.join(BASE_PATH, 'mtcnn_pytorch', 'weights', 'rnet.npy'), 
                                                      'onet': os.path.join(BASE_PATH, 'mtcnn_pytorch', 'weights', 'onet.npy')})
        smirk_weight_path = os.path.join(BASE_PATH, 'smirk/pretrained_models/SMIRK_em1.pt')
        mp_lmk_model_path = os.path.join(BASE_PATH, 'smirk/assets/face_landmarker.task')
        smirk_base_path = os.path.join(BASE_PATH, 'smirk')
        smirk_encoder = init_smirk(smirk_weight_path, device)
        flame = init_flame(smirk_base_path, device)
        renderer = init_renderer(smirk_base_path, device)
        lmk_detector = init_mp_lmk_detector(mp_lmk_model_path)

        protect_folder(mask=mask, 
                        epsilon=epsilon,
                        use_bin_mask=use_bin_mask,
                        scr_imgs_path=scr_imgs_path,
                        dst_imgs_path=dst_imgs_path,
                        device=device,
                        detector=detector,
                        img_sz=img_sz,
                        lmk_detector=lmk_detector,
                        smirk_encoder=smirk_encoder,
                        flame=flame,
                        renderer=renderer)