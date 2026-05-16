#!/usr/bin/env python3
"""Patch Pixal3D app.py for offline + Spark constraints:
- Skip the runtime utils3d wheel reinstall (we already have a source-built utils3d
  from the trellis2 base; the wheel is x86_64 only).
- Force ATTN_BACKEND to flash_attn (v2 — the only flash-attn we have on aarch64).
  The base image has flash-attn 2.7.4.post1; flash_attn_3 wheels are x86_64-only.
- Disable Gradio share tunneling so launch is fully offline.
"""
import re
import sys
from pathlib import Path

path = Path("/workspace/Pixal3D/app.py")
src = path.read_text()

# 0. Replace `import spaces` with a no-op shim so @spaces.GPU(duration=...)
#    decorators become identity functions (HF ZeroGPU is not present on Spark).
src = src.replace(
    "import spaces",
    (
        "class _FakeSpaces:\n"
        "    @staticmethod\n"
        "    def GPU(*a, **k):\n"
        "        def _d(fn): return fn\n"
        "        return _d\n"
        "spaces = _FakeSpaces()"
    ),
)

# 1. Comment out the subprocess.run that re-installs the x86_64 utils3d wheel.
src = re.sub(
    r"subprocess\.run\(\[\s*\"pip\".*?utils3d-0\.0\.2.*?\], check=True\)",
    'pass  # patched: utils3d already installed from source in base image',
    src,
    flags=re.DOTALL,
)

# 2. Force flash_attn (v2) instead of flash_attn_3.
src = src.replace(
    'os.environ["ATTN_BACKEND"] = "flash_attn_3"',
    'os.environ["ATTN_BACKEND"] = "flash_attn"',
)

# 3. Disable Gradio share tunneling on the final launch line, and drop a
#    ready-marker file once init_models() has succeeded so the entrypoint can
#    flip the offline flag for subsequent launches.
src = src.replace(
    "app.launch(show_error=True, share=True)",
    'open("/tmp/pixal3d_ready", "w").close()\n    app.launch(show_error=True, share=False)',
)

# 4. Rewrite every CDN URL in index.html to point at the locally-vendored copy
#    under /assets/vendor/ so launch is fully offline. If any CDN import fails,
#    the entire <script type="module"> aborts and the UI breaks silently.
ihtml = Path("/workspace/Pixal3D/index.html")
if ihtml.exists():
    h = ihtml.read_text()
    for cdn, local in {
        'https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js':
            '/assets/vendor/gradio-client.min.js',
        'https://unpkg.com/lucide@latest':
            '/assets/vendor/lucide.min.js',
        'https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js':
            '/assets/vendor/model-viewer.min.js',
    }.items():
        h = h.replace(cdn, local)
    ihtml.write_text(h)

path.write_text(src)

# Also patch inference.py for CLI use, same ATTN_BACKEND fix.
inf = Path("/workspace/Pixal3D/inference.py")
if inf.exists():
    s = inf.read_text().replace(
        'os.environ["ATTN_BACKEND"] = "flash_attn_3"',
        'os.environ["ATTN_BACKEND"] = "flash_attn"',
    )
    inf.write_text(s)

print("Patches applied.")
