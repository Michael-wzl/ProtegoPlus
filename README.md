# Under Development

Please do not use or distribute this code without permission from the author.

## Focal Diversity Analysis Tool

The evaluation results of size 3 and size 4 ensembles are stored in `results/eval/len3ensemble` and `results/eval/len4ensemble` respectively. Customize the focal diversity computation in `protego/focal_diversity.py`'s `get_focal_diversity` function by adding a new definition name. Use the following command to compute focal diversity and automatically analyze the results:

```bash
python3 -m tools.compute_focal_diversity --exp_name your_experiment_name --device cpu
```

You can find the analysis results in `results/focal_diversity/your_experiment_name`.

## Model Lists

- MagFace:
    - Source Code: https://github.com/IrvingMeng/MagFace
    - Paper: https://arxiv.org/abs/2103.06627 (CVPR2021)
    - Pretrained Models:
        - IR50-MagFace-MS1MV2
        - IR100-MagFace-MS1MV2
- InceptionResNetV1:
    - Source Code: https://github.com/timesler/facenet-pytorch
    - Paper: https://arxiv.org/abs/1602.07261 (AAAI2017)
    - Pretrained Models:
        - InceptionResNet-FaceNet-VGGFace2
        - InceptionResNet-FaceNet-CASIA
- Softmax-IR:
    - Source Code: https://github.com/zhongyy/OPOM
    - Paper: [Not sure which paper this corresponds to]
    - Pretrained Models:
        - IR50-Softmax-CASIA
- CosFace-IR:
    - Source Code:https://github.com/zhongyy/OPOM
    - Paper: [Not sure which paper this corresponds to]
    - Pretrained Models:
        - IR50-CosFace-CASIA
- ArcFace:
    - Source Code: https://github.com/bubbliiiing/arcface-pytorch
    - Paper: https://arxiv.org/abs/1801.07698 (CVPR2019)
    - Pretrained Models:
        - IR50-ArcFace-CASIA
        - MobileFaceNet-ArcFace-CASIA
        - MobileNet-ArcFace-CASIA
- AdaFace: 
    - Source Code: https://github.com/mk-minchul/AdaFace
    - Paper: https://arxiv.org/abs/2204.00964 (CVPR2022)
    - Pretrained Models:
        - IR18-AdaFace-WebFace
        - IR50-AdaFace-CASIA
        - IR50-AdaFace-WebFace
        - IR50-AdaFace-MS1MV2
        - IR100-AdaFace-WebFace
- ViT [Pure ViT]
    - Source Code: https://github.com/zhongyy/Face-Transformer
    - Paper: https://arxiv.org/abs/2103.14803v2 (arXiv2021)
    - Pretrained Models:
        - ViT-CosFace-MS1MV2
        - ViTs-CosFace-MS1MV2 (ViTs is the a ViT model with soft/overlapping patching)
- FaceViT [Hybrid ViT with ResNet Blocks] (didn't use in paper)
    - Source Code: https://github.com/anguyen8/face-vit
    - Paper: https://arxiv.org/abs/2311.02803 (WACV2023)
    - Pretrained Models:
        - FaceViT-ArcFace-WebFace (Cross-image attention model)
- PartfViT [Hybrid ViT with patches based on the face landmarks given by MobileNetV3] (didn't use in paper)
    - Source Code: https://github.com/szlbiubiubiu/LAFS_CVPR2024
    - Paper: https://arxiv.org/abs/2403.08161 (CVPR2024)
    - Pretrained Models:
        - PartfViT-CosFace-MS1MV3
        - PartfViT-ArcFace-WebFace
- TransFace [Pure ViT with patchify done by convolution and weighing patches with SENet-style module] (didn't use in paper)
    - Source Code: https://github.com/DanJun6737/TransFace
    - Paper: https://openaccess.thecvf.com/content/ICCV2023/html/Dan_TransFace_Calibrating_Transformer_Training_for_Face_Recognition_from_a_Data-Centric_ICCV_2023_paper.html (ICCV2023)
    - Pretrained
        - TransFace-S-ArcFace-MS1MV2
        - TransFace-B-ArcFace-MS1MV2 (bigger model)
        - TransFace-L-ArcFace-MS1MV2 (largest model)
        - TransFace-S-ArcFace-Glint360k
        - TransFace-B-ArcFace-Glint360k (bigger model)
        - TransFace-L-ArcFace-Glint360k (largest model)
- SwinFace [Pure ViT but with CNN-style hierarchical architecture and patch merging] (didn't use in paper)
    - Source Code: https://github.com/lxq1000/SwinFace
    - Paper: https://arxiv.org/pdf/2308.11509.pdf (TCSVT2024)
    - Pretrained Models:
        - SwinFace-T-ArcFace-MS1MV2
