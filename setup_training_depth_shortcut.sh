#!/bin/bash
# =============================================================
# Self-Flow MDM Depth Shortcut Training - Full Setup Script
# Tested target: Python 3.10-3.12, CUDA 11.x / 12.x, Ubuntu 20.04+
#
# Usage:
#   HF_TOKEN=... bash setup_training_depth_shortcut.sh [WORKDIR]
#
# WORKDIR defaults to /workspace.
# Override repo source if needed:
#   REPO_URL=https://github.com/<user>/Self-Flow.git BRANCH=clone/motion-diffusion-model bash setup_training_depth_shortcut.sh
# =============================================================
set -euo pipefail

WORKDIR="${1:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/thanhlamauto/Self-Flow.git}"
BRANCH="${BRANCH:-clone/motion-diffusion-model}"
REPO_DIR="$WORKDIR/Self-Flow"

# -- colors ----------------------------------------------------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[SETUP]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN] ${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

info "Working directory : $WORKDIR"
info "Repo directory    : $REPO_DIR"
info "Repo URL          : $REPO_URL"
info "Branch            : $BRANCH"

# =============================================================
# 1. System packages
# =============================================================
info "Installing system packages..."
apt-get update -y -qq
apt-get install -y -qq \
    git wget curl unzip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0

# =============================================================
# 2. Clone or update repo
# =============================================================
info "Preparing Self-Flow repo..."
mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [ -d "$REPO_DIR/.git" ]; then
    warn "Repo already exists at $REPO_DIR."
    cd "$REPO_DIR"
    git fetch origin "$BRANCH"
    git checkout "$BRANCH"
    git pull --ff-only origin "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# =============================================================
# 3. Install PyTorch (auto-detect CUDA version)
# =============================================================
info "Detecting CUDA version..."

CUDA_MAJOR=$(nvidia-smi 2>/dev/null \
    | grep "CUDA Version" \
    | grep -oP 'CUDA Version: \K[0-9]+' \
    || nvcc --version 2>/dev/null \
    | grep -oP 'release \K[0-9]+' \
    || echo "cpu")

if [ "$CUDA_MAJOR" = "cpu" ]; then
    warn "No GPU/CUDA detected; installing CPU-only PyTorch."
    pip install --quiet torch torchvision torchaudio
elif [ "$CUDA_MAJOR" -ge 12 ]; then
    info "CUDA $CUDA_MAJOR detected; installing PyTorch cu124 wheels."
    pip install --quiet torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
else
    info "CUDA $CUDA_MAJOR detected; installing PyTorch cu118 wheels."
    pip install --quiet torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu118
fi

# =============================================================
# 4. Python dependencies
# =============================================================
info "Installing Python packages..."
pip install --quiet \
    numpy \
    scipy \
    spacy \
    "smplx==0.1.28" \
    "chumpy==0.70" \
    six \
    joblib \
    scikit-learn \
    "moviepy==1.0.3" \
    wandb \
    gdown \
    "huggingface_hub>=0.20" \
    ftfy \
    regex \
    tqdm \
    imageio \
    blobfile \
    matplotlib

info "Downloading spaCy English model..."
python -m spacy download en_core_web_sm -q

info "Installing CLIP..."
mkdir -p sub_modules
if [ ! -d "sub_modules/CLIP/.git" ]; then
    git clone --quiet https://github.com/openai/CLIP.git sub_modules/CLIP
fi
pip install --quiet -e sub_modules/CLIP

# =============================================================
# 5. Compatibility patches
# =============================================================
info "Patching chumpy for NumPy 2.x / Python 3.12 compatibility..."
python - <<'PYEOF'
import os

pkg_dir = os.path.dirname(__import__("chumpy").__file__)
init_path = os.path.join(pkg_dir, "__init__.py")

with open(init_path) as f:
    content = f.read()

old = "from numpy import bool, int, float, complex, object, unicode, str, nan, inf"
new = "from numpy import nan, inf"

if old in content:
    with open(init_path, "w") as f:
        f.write(content.replace(old, new))
    print("  chumpy __init__.py patched")
else:
    print("  chumpy __init__.py already clean")
PYEOF

cat > "$REPO_DIR/sitecustomize.py" <<'EOF'
import inspect
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
EOF
info "sitecustomize.py written"

# =============================================================
# 6. Download model/eval assets
# =============================================================
info "Downloading SMPL body model..."
if [ -f "body_models/smpl/SMPL_NEUTRAL.pkl" ]; then
    warn "SMPL already present; skipping."
else
    bash prepare/download_smpl_files.sh
fi

info "Downloading T2M evaluators..."
if [ -f "t2m/Comp_v6_KLD01/model/latest.tar" ]; then
    warn "T2M evaluators already present; skipping."
else
    bash prepare/download_t2m_evaluators.sh
fi

info "Downloading GloVe embeddings..."
if [ -f "glove/our_vab_data.npy" ]; then
    warn "GloVe already present; skipping."
else
    bash prepare/download_glove.sh
fi

# =============================================================
# 7. Download HumanML3D dataset from HuggingFace
# =============================================================
info "Downloading HumanML3D dataset from HuggingFace..."

if [ -d "datasets/HumanML3D/new_joint_vecs" ]; then
    warn "HumanML3D dataset already present; skipping."
else
    if [ -z "${HF_TOKEN:-}" ]; then
        error "HF_TOKEN is required to download HumanML3D. Re-run with: HF_TOKEN=... bash setup_training_depth_shortcut.sh"
    fi
    mkdir -p datasets/HumanML3D
    python - <<PYEOF
from huggingface_hub import snapshot_download

print("  Connecting to HuggingFace...")
snapshot_download(
    repo_id="Namthukhoa214/HumanML3D",
    repo_type="dataset",
    token="${HF_TOKEN}",
    local_dir="datasets/HumanML3D",
    ignore_patterns=["*.git*", ".gitattributes"],
)
print("  Dataset download complete.")
PYEOF
fi

info "Creating dataset symlink..."
mkdir -p dataset
ln -sfn "$REPO_DIR/datasets/HumanML3D" "$REPO_DIR/dataset/HumanML3D"
info "  dataset/HumanML3D -> datasets/HumanML3D"

# =============================================================
# 8. Verify installation
# =============================================================
info "Verifying installation..."

python - <<'PYEOF'
errors = []

try:
    import torch
    print(f"  torch       {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    errors.append(f"torch: {e}")

try:
    import chumpy
    print("  chumpy      OK")
except Exception as e:
    errors.append(f"chumpy: {e}")

try:
    import clip
    print("  CLIP        OK")
except ImportError as e:
    errors.append(f"CLIP: {e}")

try:
    import smplx
    print(f"  smplx       {smplx.__version__}")
except ImportError as e:
    errors.append(f"smplx: {e}")

try:
    import spacy
    spacy.load("en_core_web_sm")
    print(f"  spacy       {spacy.__version__} | en_core_web_sm OK")
except Exception as e:
    errors.append(f"spacy: {e}")

import os
checks = {
    "SMPL neutral model": "body_models/smpl/SMPL_NEUTRAL.pkl",
    "T2M evaluator": "t2m/Comp_v6_KLD01/model/latest.tar",
    "GloVe embeddings": "glove/our_vab_data.npy",
    "HumanML3D vecs": "datasets/HumanML3D/new_joint_vecs",
    "HumanML3D texts": "datasets/HumanML3D/texts",
    "dataset symlink": "dataset/HumanML3D",
}
for name, path in checks.items():
    exists = os.path.exists(path)
    status = "OK" if exists else "MISSING"
    marker = "  " if exists else "!!"
    print(f"  {marker} {name:30s} {status} ({path})")
    if not exists:
        errors.append(f"Missing: {path}")

if errors:
    print("\nErrors found:")
    for e in errors:
        print(f"  - {e}")
    raise SystemExit(1)

print("\nAll checks passed!")
PYEOF

# =============================================================
# Done - print training command
# =============================================================
echo ""
echo -e "${GREEN}=============================================================${NC}"
echo -e "${GREEN} Setup complete. Run depth-shortcut MDM training with:${NC}"
echo -e "${GREEN}=============================================================${NC}"
echo ""
echo "cd $REPO_DIR && DATA_DIR=$REPO_DIR/dataset/HumanML3D NPROC=4 bash scripts/train_humanml_depth_shortcut_4x4090.sh"
echo ""
