#!/bin/bash
set -e

echo "--- Setting up phi4-analyze environment ---"

git config --global user.name "Teodor"
git config --global user.email "teo@example.com"

VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate
pip install --no-cache-dir transformers kernels torch accelerate bitsandbytes huggingface_hub safetensors tqdm jupyter matplotlib seaborn pandas ipython ipykernel torchinfo
python -m ipykernel install --user --name=venv --display-name "Python (Odor_venv)"
deactivate

echo "Setup complete. Run:"
echo "  source venv/bin/activate"
echo "Then (e.g.):"
echo "  python analyze_phi4.py"