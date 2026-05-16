#!/bin/bash
set -e

echo "========================================"
echo "Pixal3D — Spark AI Hub recipe"
echo "========================================"

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
export PATH="$CUDA_HOME/bin:${PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "========================================"

# First-run pre-warm: download Pixal3D + MoGe + DinoV3 weights to the mounted
# HF cache volume. Forces online mode for this one step so that subsequent
# launches can run with HF_HUB_OFFLINE=1.
PREWARM_MARKER="${HF_HOME:-/workspace/cache/huggingface}/.pixal3d_prewarmed"
if [ ! -f "$PREWARM_MARKER" ]; then
  echo "First launch detected — pre-downloading model weights..."
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
for repo in [
    "TencentARC/Pixal3D-T",
    "camenduru/dinov3-vitl16-pretrain-lvd1689m",
    "Ruicheng/moge-2-vitl",
]:
    print(f"Fetching {repo}...")
    snapshot_download(repo_id=repo, token=os.environ.get("HF_TOKEN") or None)
print("Pre-warm complete.")
PYEOF
  touch "$PREWARM_MARKER"
fi

# Same RMBG patch as trellis2 base (PyTorch nightly meta-tensor incompat)
find "${HF_HOME:-/workspace/cache/huggingface}" -path "*/RMBG*2*/birefnet.py" -exec \
  sed -i 's/\[x\.item() for x in torch\.linspace(0, drop_path_rate, sum(depths))\]/[float(x) for x in __import__("numpy").linspace(0, drop_path_rate, sum(depths))]/g' {} + 2>/dev/null || true

if [ $# -eq 0 ]; then
  exec python app_local.py
else
  exec "$@"
fi
