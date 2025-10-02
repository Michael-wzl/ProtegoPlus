import functools

from .transform import (get_crop_and_resize_matrix, get_face_align_matrix, get_face_align_matrix_celebm,
                        make_inverted_tanh_warp_grid, make_tanh_warp_grid)

pretrain_settings = {
    'lapa/448': {
        'url': [
            'https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.lapa.main_ema_136500_jit191.pt',
        ],
        'matrix_src_tag': 'points',
        'get_matrix_fn': functools.partial(get_face_align_matrix,
                                           target_shape=(448, 448), target_face_scale=1.0),
        'get_grid_fn': functools.partial(make_tanh_warp_grid,
                                         warp_factor=0.8, warped_shape=(448, 448)),
        'get_inv_grid_fn': functools.partial(make_inverted_tanh_warp_grid,
                                             warp_factor=0.8, warped_shape=(448, 448)),
        'label_names': ['background', 'face', 'rb', 'lb', 're',
                        'le', 'nose',  'ulip', 'imouth', 'llip', 'hair']
    },
    'celebm/448': {
        'url': [
            'https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.celebm.main_ema_181500_jit.pt',
        ],
        'matrix_src_tag': 'points',
        'get_matrix_fn': functools.partial(get_face_align_matrix_celebm,
                                           target_shape=(448, 448)),
        'get_grid_fn': functools.partial(make_tanh_warp_grid,
                                         warp_factor=0, warped_shape=(448, 448)),
        'get_inv_grid_fn': functools.partial(make_inverted_tanh_warp_grid,
                                             warp_factor=0, warped_shape=(448, 448)),
        'label_names':  [
                    'background', 'neck', 'face', 'cloth', 'rr', 'lr', 'rb', 'lb', 're',
                    'le', 'nose', 'imouth', 'llip', 'ulip', 'hair',
                    'eyeg', 'hat', 'earr', 'neck_l']
    }
}