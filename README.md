# Protego: User-Centric Pose-Invariant Privacy Protection Against Face Recognition-Induced Digital Footprint Exposure

## Environment Setup

1. The code has been tested on the following machines. Environments close to these are preferable. Windows is not recommended.

    - Ubuntu 24.04 x86_64; AMD EPYC 7282; NVIDIA RTX 4090 (CUDA ≥ **11.7**); 256 GB RAM
    - Ubuntu 22.04 x86_64; Intel Xeon w5-3415; NVIDIA RTX 5880 Ada (CUDA ≥ **11.7**); 128 GB RAM
    - MacOS; Apple Silicon (M4 Pro); MPS backend; 24 GB RAM
    - MacOS; Apple Silicon (M3); MPS backend; 16 GB RAM

2. Make sure you have the following commands available in your terminal before setting up the environment:

    - git
    - wget
    - conda (Miniforge3 is preferred, tested)

3. Make sure you haven't already created a conda environment named `protego` (or the name you will use in the setup script) to avoid conflicts.

4. We need to download FLAME assets, which require registration at [FLAME](https://flame.is.tue.mpg.de/). You will be prompted for your username and password during the setup and the asset will be downloaded automatically.

5. Run the following commands. The bash script will create a conda environment named `protego` (you may change it in `setup_quick.sh`) with Python **3.9**, install all the necessary libraries, and download necessary assets. It will adapt to your machine automatically.

    ```bash
    bash setup_quick.sh && conda activate protego
    ```

## Download Necessary Weights Manually

Some weights cannot downloaded through scripts and must thus be manually downloaded.

1. ArcFace Models

Download the following models from the [Baidu Netdisk links](https://pan.baidu.com/s/1ElJlfmMwOGX699MsgLY8qA) (Pass Code: z3rq) provided by the [PyTorch implementation of ArcFace](https://github.com/bubbliiiing/arcface-pytorch) and put the 3 weights into the `FR_DB/arcface/pretrained` folder. (You may want to translate the page to English and find the link in the "File Download" section.)

- arcface_iresnet50.pth
- arcface_mobilenet_v1.pth
- arcface_mobilefacenet.pth

## Download and Preprocess FaceScrub Dataset

1. Download the FaceScrub dataset from [Kaggle](https://www.kaggle.com/datasets/rajnishe/facescrub-full).

2. Unzip the downloaded file and put the unzipped folder under `face_db/` and rename it to `face_scrub`. At this point, the structure should be like:

    ```plaintext
    face_db/
    ├── face_scrub/
    │   ├── actor_faces/
    │   │   ├── Aaron_Eckhart/
    │   │   └── ...
    │   └── actress_faces/
    │       ├── Adrianne_Leo╠ün/
    │       └── ...
    └── ...
    ```

3. Run the following command to reorganize and preprocess the dataset. Note that pretrained weights of face recognition models will be downloaded automatically in the first run, so stable internet connection is required.

    ```bash
    bash reorganize_fs.sh
    python3 -m tools.build_face_db --device cuda:0
    ```

## Train PPTs

Train privacy protection textures (PPTs) by running the following command with default settings. (Change `--exp_name` and `--device` as needed.) You may change other training parameters in `train_ppt.py` (see the file for details).

```bash
python3 train_ppt.py --exp_name default --device cuda:0
```

After training, you may find the trained PPTs (`univ_mask.npy`), visualizations, and default evaluation results under `experiments/default/`. You can visualize PPTs' default evaluation results with the following command. The plot will be saved to `experiments/default/ir50_adaface_casia_evaluation_results.png`.

```bash
python3 -m tools.analyze_res --exp_name default
```

## Evaluate PPTs

Evaluate the PPTs trained during the `default` experiment under more diverse setting by specifying parameters in `eval_ppt.py` (see the file for details). For example, to evaluate the default PPTs' robustness against different compression methods and `IR50-AdaFace-CASIA` with the following command. The evaluation results will be saved under `results/eval/default/`.

```bash
python3 eval_ppt.py --mask_name default --exp_name compression_robustness --device cuda:0
```

You can visualize the evaluation results with the following command. The plot will be saved to `results/eval/default/ir50_adaface_casia_compression_results.png`. For visualizing results of other settings, change the flags as needed. See `tools/analyze_res.py` for details.

```bash
python3 -m tools.analyze_res --eval_exp_name compression_robustness --compression --need_lpips
```

## Apply PPTs to Images

You can apply the PPTs trained during the `default` experiment to images of Bradley Cooper by putting your images under `face_db/imgs/Bradley_Cooper/` and running the following command:

```bash
python3 -m tools.protect_imgs --mask_name default --protectee Bradley_Cooper --device cuda:0
```

You may protect others' images by placing their images under `face_db/imgs/{protectee}/` and changing the `--protectee` flag accordingly. However, note that PPTs are person-specific and thus need prior training on the protectee. For simplicity, only use images of the selected 20 protectees in `face_db/face_scrub/`.

## Applying PPTs to Videos

You can apply the PPTs trained during the `default` experiment to videos of Hugh Grant by putting videos under `face_db/vids/Hugh_Grant/` and running the following command:

```bash
python3 -m tools.protect_vids --mask_name default --protectee Hugh_Grant --vid_name hg1.mp4 --device cuda:0
```

You may protect others' videos by placing their videos under `face_db/vids/{protectee}/` and changing the `--protectee` flag accordingly. However, note that PPTs are person-specific and thus need prior training on the protectee. For simplicity, only use videos of the selected 20 protectees in `face_db/face_scrub/`. Also, for simplicity, our current implementation expects the video to be a speech with a single face appearing constantly throughout the video. A good example is this speech video of [Hugh Grant](https://www.youtube.com/watch?v=gugpWvACiiw).

## Acknowledgements

All codes in `FR_DB`, `FD_DB/RetinaFace`, `DiffJPEG`, and `smirk` are adapted from the following repositories:

- `FR_DB`
  - [adaface](https://github.com/mk-minchul/AdaFace)
  - [arcface](https://github.com/bubbliiiing/arcface-pytorch)
  - [facenet](https://github.com/timesler/facenet-pytorch)
  - [ir50_opom](https://github.com/zhongyy/OPOM)
  - [magface](https://github.com/IrvingMeng/MagFace)
  - [swinface](https://github.com/lxq1000/SwinFace)
  - [vit]( https://github.com/zhongyy/Face-Transformer)
- `FD_DB/RetinaFace`
  - [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface)
- `DiffJPEG`
  - [DiffJPEG](https://github.com/mlomnitz/DiffJPEG)
- `smirk`
  - [smirk](https://github.com/georgeretsi/smirk)
