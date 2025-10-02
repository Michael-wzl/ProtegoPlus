import functools

import torch
from .transform import get_face_align_matrix, make_tanh_warp_grid
import numpy as np

def get_std_points_xray(out_size:int = 256, mid_size:int = 500) -> torch.Tensor:
    std_points_256 = np.array(
        [
            [85.82991, 85.7792],
            [169.0532, 84.3381],
            [127.574, 137.0006],
            [90.6964, 174.7014],
            [167.3069, 173.3733],
        ]
    )
    std_points_256[:, 1] += 30
    old_size = 256
    mid = mid_size / 2
    new_std_points = std_points_256 - old_size / 2 + mid
    target_pts = new_std_points * out_size / mid_size
    target_pts = torch.from_numpy(target_pts).float()
    return target_pts

pretrain_settings = {
    "celeba/224": {
        # acc 92.06617474555969
        "num_classes": 40,
        "layers": [11],
        "url": "https://github.com/FacePerceiver/facer/releases/download/models-v1/face_attribute.farl.celeba.pt",
        "matrix_src_tag": "points",
        "get_matrix_fn": functools.partial(
            get_face_align_matrix,
            target_shape=(224, 224),
            target_pts=get_std_points_xray(out_size=224, mid_size=500),
        ),
        "get_grid_fn": functools.partial(
            make_tanh_warp_grid, warp_factor=0.0, warped_shape=(224, 224)
        ),
        "classes": [
            "5_o_Clock_Shadow",
            "Arched_Eyebrows",
            "Attractive",
            "Bags_Under_Eyes",
            "Bald",
            "Bangs",
            "Big_Lips",
            "Big_Nose",
            "Black_Hair",
            "Blond_Hair",
            "Blurry",
            "Brown_Hair",
            "Bushy_Eyebrows",
            "Chubby",
            "Double_Chin",
            "Eyeglasses",
            "Goatee",
            "Gray_Hair",
            "Heavy_Makeup",
            "High_Cheekbones",
            "Male",
            "Mouth_Slightly_Open",
            "Mustache",
            "Narrow_Eyes",
            "No_Beard",
            "Oval_Face",
            "Pale_Skin",
            "Pointy_Nose",
            "Receding_Hairline",
            "Rosy_Cheeks",
            "Sideburns",
            "Smiling",
            "Straight_Hair",
            "Wavy_Hair",
            "Wearing_Earrings",
            "Wearing_Hat",
            "Wearing_Lipstick",
            "Wearing_Necklace",
            "Wearing_Necktie",
            "Young",
        ],
    }
}