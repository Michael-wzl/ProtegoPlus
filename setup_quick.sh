#!/bin/bash
OS_TYPE=$(uname)
if [[ "$OS_TYPE" == "Darwin" ]]; then
    echo "Running on macOS"
    MACOS_VER=$(sw_vers -productVersion | awk -F '.' '{print $1"."$2}')
elif [[ "$OS_TYPE" == "Linux" ]]; then
    echo "Running on Linux"
else
    echo "Unsupported OS: $OS_TYPE"
    exit 1
fi

check_cuda_support() {
    if command -v lspci &> /dev/null; then
        if lspci | grep -i nvidia &> /dev/null; then
            echo "NVIDIA GPU detected. CUDA might be supported."
            return 0
        else
            echo "No NVIDIA GPU detected. CUDA is not supported."
            return 1
        fi
    else
        echo "lspci command not found. Unable to check for NVIDIA GPU."
        return 1
    fi
}
CUDA_SUPPORT=0
if [[ "$OS_TYPE" == "Linux" ]]; then
    check_cuda_support
    CUDA_SUPPORT=$?
    if [[ $CUDA_SUPPORT -ne 0 ]]; then
        echo "CUDA is not supported on this machine. Exiting." 
        exit 1
    fi
fi

echo "Downloading SMIRK assets..."
mkdir tmp && cd tmp
git clone https://github.com/georgeretsi/smirk
mv smirk/assets ../smirk/
cd ..
rm -rf tmp

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate base

ENV_NAME="protego_plus"
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
echo "Installing packages and downloading SMIRK weights..."
if [[ "$OS_TYPE" == "Linux" ]]; then
    pip install -r requirements.txt
elif [[ "$OS_TYPE" == "Darwin" ]]; then
    pip install -r requirements_mac.txt
fi
conda install zip -y
conda install unzip -y
if [[ "$OS_TYPE" == "Linux" ]]; then
    pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt201/download.html
elif [[ "$OS_TYPE" == "Darwin" ]]; then
    git clone https://github.com/facebookresearch/pytorch3d.git
    cd pytorch3d
    MACOSX_DEPLOYMENT_TARGET=$MACOS_VER CC=clang CXX=clang++ pip install . 
    cd ..
    rm -rf pytorch3d
fi
cd smirk
bash quick_install.sh
pip install pytorch_msssim==1.0.0
conda install requests=2.32.3 -y
conda install termcolor=3.1.0 -y
conda install ipython=8.18.1 -y
conda install ipykernel -y
pip install einops 
pip install thop
pip install lpips
pip install kornia
pip install av
conda install -c conda-forge ffmpeg -y
pip install cvxpy
cd ..
echo "All packages installed successfully!"

echo "Downloading MTCNN weights..."
cd FD_DB/MTCNN/pretrained/
gdown --fuzzy "https://drive.google.com/file/d/1uJopXpkHHzzImZ-4LVWrRHHMbUECi5Fb/view?usp=share_link"
unzip mtcnn_pytorch_weights.zip
rm -f mtcnn_pytorch_weights.zip
cd ../../..

#echo "Downloading IR50-AdaFace-CASIA weights..."
#cd FR_DB/adaface/pretrained
#gdown --fuzzy "https://drive.google.com/file/d/1g1qdg7_HSzkue7_VrW64fnWuHl0YL2C2/view?usp=sharing"
#cd ../../..

#echo "Downloading Processed FaceScrub Dataset..."
cd face_db
#gdown --fuzzy "https://drive.google.com/file/d/1_Mfq10d1fdDJGDum4QKZphHNXEc8LT0J/view?usp=sharing" # TODO Updated link, with imgs_list.txt
#unzip face_scrub_preprocessed.zip
#rm -f face_scrub_preprocessed.zip

echo "Downloading Demo Video and Images..."
gdown --fuzzy "https://drive.google.com/file/d/14U8zeWsgqrduJ5wr0l5Iv24-Vd1fZqZs/view?usp=sharing"
unzip demo_vids_bradley_cooper.zip
rm -f demo_vids_bradley_cooper.zip
gdown --fuzzy "https://drive.google.com/file/d/1SmnKTaPw82hjWcgf-td921licl-BBowD/view?usp=sharing"
unzip demo_imgs_bradley_cooper.zip
rm -f demo_imgs_bradley_cooper.zip
cd ..

echo "Downloading Pretrained Pose-invariant PPTs..."
cd experiments
gdown --fuzzy "https://drive.google.com/file/d/1SymmnmEebg_DfUSSrn446Le38eeycetl/view?usp=share_link"
unzip default.zip
rm -f default.zip
cd ..

# ! Temporary for Focal Diversity Analysis
#echo "Downloading Evaluation Results for Focal Diversity Analysis..."
#cd results && mkdir eval && cd eval
#gdown --fuzzy "https://drive.google.com/file/d/1hTd8DGrDdEAZvZNMMxfAiTK2Sn5IipwX/view?usp=sharing" # len3ensemble.zip
#unzip len3ensemble.zip
#rm -f len3ensemble.zip
#gdown --fuzzy "https://drive.google.com/file/d/1m1r7mlxOHSiGbIkg7VFWrystUCjfLSqH/view?usp=sharing" # len4ensemble.zip
#unzip len4ensemble.zip
#rm -f len4ensemble.zip
#cd ../..

echo "ALL DONE!!!"
