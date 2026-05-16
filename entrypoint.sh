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

# First-launch model fetch: any HF asset Pixal3D needs is downloaded the first
# time the container comes up, then cached in the mounted volume. Subsequent
# launches respect HF_HUB_OFFLINE=1 from the compose env and stay fully offline.
PREWARM_MARKER="${HF_HOME:-/workspace/cache/huggingface}/.pixal3d_prewarmed"
if [ ! -f "$PREWARM_MARKER" ]; then
  echo "First launch — enabling HF online mode so init_models() can fetch all assets."
  export HF_HUB_OFFLINE=0
  export TRANSFORMERS_OFFLINE=0
  # Pre-bulk the three main repos to surface DNS/auth errors early.
  python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
for repo in [
    "TencentARC/Pixal3D-T",
    "camenduru/dinov3-vitl16-pretrain-lvd1689m",
    "Ruicheng/moge-2-vitl",
]:
    print(f"Fetching {repo}...")
    snapshot_download(repo_id=repo, token=os.environ.get("HF_TOKEN") or None)
print("Bulk pre-fetch complete; remaining assets will be pulled by init_models().")
PYEOF
  # Background watcher: as soon as app.py creates /tmp/pixal3d_ready (i.e.
  # init_models() succeeded), write the persistent prewarm marker so that
  # subsequent container starts respect HF_HUB_OFFLINE=1 from the compose env.
  (
    while [ ! -f /tmp/pixal3d_ready ]; do sleep 5; done
    touch "$PREWARM_MARKER"
    echo "Prewarm marker written — future launches will run offline."
  ) &
fi

# Same RMBG patch as trellis2 base (PyTorch nightly meta-tensor incompat)
find "${HF_HOME:-/workspace/cache/huggingface}" -path "*/RMBG*2*/birefnet.py" -exec \
  sed -i 's/\[x\.item() for x in torch\.linspace(0, drop_path_rate, sum(depths))\]/[float(x) for x in __import__("numpy").linspace(0, drop_path_rate, sum(depths))]/g' {} + 2>/dev/null || true

if [ $# -eq 0 ]; then
  exec python app.py
else
  exec "$@"
fi
