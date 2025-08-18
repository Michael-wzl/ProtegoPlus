#!/bin/bash

echo "Downloading SMIRK assets and MTCNN weights..."
mkdir tmp && cd tmp
git clone https://github.com/georgeretsi/smirk
mv smirk/assets ../smirk/
git clone https://github.com/TropComplique/mtcnn-pytorch
mv mtcnn-pytorch/src/weights/*.npy ../mtcnn_pytorch/src/weights
cd .. && rm -rf tmp

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate base

ENV_NAME="protego"
PYTHON_VERSION="3.9" 
if conda env list | grep -q "^$ENV_NAME\s"; then
    echo "Error: Conda environment '$ENV_NAME' already exists. Please remove it or choose a different name."
    exit 1
fi
echo "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
echo "Activating conda environment: $ENV_NAME..."
conda activate $ENV_NAME
check_target_env() {
    if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
        echo "Error: Attempting to install packages to '$CONDA_DEFAULT_ENV'. Please remove the automatically created env and set up the environment manually."
        exit 1
    fi
}
check_target_env
echo "Installing packages..."
pip install -r requirements.txt
conda install zip -y
conda install unzip -y
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt201/download.html
cd smirk
bash quick_install.sh
pip install pytorch_msssim==1.0.0
conda install requests=2.32.3 -y
conda install termcolor=3.1.0 -y
conda install ipython=8.18.1 -y
conda install ipykernel -y
cd ..
echo "All packages installed successfully!"

echo "Downloading IR50-AdaFace-CASIA weights..."
cd FR_DB/adaface/pretrained
gdown --fuzzy https://drive.google.com/file/d/1g1qdg7_HSzkue7_VrW64fnWuHl0YL2C2/view?usp=sharing
cd ../../..

echo "Downloading Processed FaceScrub Dataset..."
cd face_db
gdown --fuzzy https://drive.google.com/file/d/1H-exPEZXCKb8hP7SBcYSLeqWxhtrrbma/view?usp=sharing
unzip face_scrub_preprocessed.zip
rm face_scrub_preprocessed.zip

echo "Downloading Demo Video and Images..."
gdown --fuzzy https://drive.google.com/file/d/14U8zeWsgqrduJ5wr0l5Iv24-Vd1fZqZs/view?usp=sharing
unzip demo_vids_bradley_cooper.zip
rm -f demo_vids_bradley_cooper.zip
gdown --fuzzy https://drive.google.com/file/d/1SmnKTaPw82hjWcgf-td921licl-BBowD/view?usp=sharing
unzip demo_imgs_bradley_cooper.zip
rm -f demo_imgs_bradley_cooper.zip
cd ..

echo "Downloading Pretrained Pose-invariant PPTs..."
gdown --fuzzy https://drive.google.com/file/d/1Ckh9To_EoUhcwooJ6con3DGtkVyVdGSu/view?usp=sharing
unzip pretrained_ppts_protego.zip
rm -f pretrained_ppts_protego.zip
cd ..

echo "ALL DONE!!!"
