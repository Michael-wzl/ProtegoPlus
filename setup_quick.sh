#!/bin/bash
####################################################################################################################
# Configuration
ENV_NAME="protego_cvpr"
PYTHON_VERSION="3.9" 
####################################################################################################################

# Check OS type
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

# Check CUDA support for Linux
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

# Download SMIRK assets
echo "Downloading SMIRK assets..."
mkdir tmp && cd tmp
git clone https://github.com/georgeretsi/smirk
mv smirk/assets ../smirk/
cd ..
rm -rf tmp

# Create and activate conda environment
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate base

if conda env list | grep -q "^$ENV_NAME\s"; then
    echo "Error: Conda environment '$ENV_NAME' already exists. Please remove it or choose a different name."
    exit 1
fi
echo "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
echo "Activating conda environment: $ENV_NAME..."
conda activate $ENV_NAME
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "Error: Attempting to install packages to '$CONDA_DEFAULT_ENV', which isn't the target env. Please remove the automatically created env and set up the environment manually."
    exit 1
fi
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

# Download MTCNN weights
echo "Downloading MTCNN weights..."
mkdir tmp && cd tmp
git clone https://github.com/TropComplique/mtcnn-pytorch
cd ..
mv tmp/mtcnn-pytorch/src/weights/* FD_DB/MTCNN/pretrained/
rm -rf tmp

# Donwload IR50-OPOM weights
echo "Downloading IR50-OPOM weights..."
cd FR_DB/ir50_opom/pretrained
gdown --fuzzy https://drive.google.com/file/d/1XmHD2mTcc6SHutCVPVw7cVg5jGkGIIUU/view?usp=sharing
unzip models.zip && rm -f models.zip
mv models/surrogate/IR_50-CosFace-casia/* .
mv models/surrogate/IR_50-Softmax-casia/* .
rm -rf models
cd ../../..

# Create file structures
cd face_db
mkdir imgs vids
cd ..
cd results
mkdir imgs vids eval
cd ..

echo "ALL DONE!!!"
