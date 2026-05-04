#!/usr/bin/env bash
set -euo pipefail

# VM bootstrap for HumanML3D MDM training on 4xRTX4090.
# Assumes Ubuntu, NVIDIA driver already installed, and conda/mamba available.

ENV_NAME="${ENV_NAME:-mdm4090}"
DATA_DIR="${DATA_DIR:-$PWD/dataset/HumanML3D}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniconda/Miniforge first, then rerun this script." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y git wget curl unzip unrar ffmpeg libgl1 libglib2.0-0 build-essential

conda create -y -n "${ENV_NAME}" python=3.10
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  blobfile==2.1.1 \
  chumpy==0.70 \
  einops \
  ftfy \
  gdown \
  h5py \
  huggingface_hub \
  matplotlib \
  numpy==1.23.5 \
  pandas \
  regex \
  scikit-learn \
  scipy \
  smplx==0.1.28 \
  spacy==3.7.4 \
  tqdm \
  trimesh \
  wandb
python -m pip install git+https://github.com/openai/CLIP.git
python -m spacy download en_core_web_sm

bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
bash prepare/download_smpl_files.sh

mkdir -p "${DATA_DIR}"
huggingface-cli download Namthukhoa214/HumanML3D \
  --repo-type dataset \
  --local-dir "${DATA_DIR}" \
  --local-dir-use-symlinks False

if [ -f "${DATA_DIR}/texts.tar.gz" ] && [ ! -d "${DATA_DIR}/texts" ]; then
  tar -xzf "${DATA_DIR}/texts.tar.gz" -C "${DATA_DIR}"
fi

if [ -f "${DATA_DIR}/new_joint_vecs.rar" ] && [ ! -d "${DATA_DIR}/new_joint_vecs" ]; then
  unrar x -o+ "${DATA_DIR}/new_joint_vecs.rar" "${DATA_DIR}/"
fi

echo "Setup complete."
echo "Activate with: conda activate ${ENV_NAME}"
echo "HumanML3D path: ${DATA_DIR}"
