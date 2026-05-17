#!/bin/bash
# Self-test: drive the Pixal3D Gradio API end-to-end so we don't have to
# bother the user every iteration. Runs inside the container.
set -e
docker exec spark-ai-hub-pixal3d python - <<'PYEOF'
import sys, traceback
from gradio_client import Client, handle_file
c = Client("http://127.0.0.1:7860")
img = "/workspace/Pixal3D/assets/images/0_img.png"
print("[selftest] /preprocess ...", flush=True)
pre = c.predict(image=handle_file(img), api_name="/preprocess")
print("[selftest] preprocess OK ->", pre, flush=True)
print("[selftest] /generate_3d ...", flush=True)
try:
    out = c.predict(
        image=handle_file(pre),
        seed=42, resolution=1024,
        api_name="/generate_3d",
    )
    print("[selftest] generate_3d OK keys:", list(out.keys()) if isinstance(out, dict) else type(out), flush=True)
except Exception as e:
    print("[selftest] generate_3d FAILED:", e, flush=True)
    traceback.print_exc()
    sys.exit(1)
PYEOF
