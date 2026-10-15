#!/usr/bin/env python3
"""Patch Pixal3D app.py for offline + Spark constraints:
- Skip the runtime utils3d wheel reinstall (we already have a source-built utils3d
  from the trellis2 base; the wheel is x86_64 only).
- Force ATTN_BACKEND to flash_attn (v2 — the only flash-attn we have on aarch64).
  The base image has flash-attn 2.7.4.post1; flash_attn_3 wheels are x86_64-only.
- Disable Gradio share tunneling so launch is fully offline.
- Load DINOv3 from Meta's own repo instead of the camenduru re-upload upstream
  names (byte-identical model.safetensors; Meta's repo is gated, manual approval).
- Make every URL in index.html relative and serve its static files at
  /px-assets, so the page works behind the Hub's /run/pixal3d-spark/ prefix.
- Write GLBs through glb_export.py: the front on +Z (so the web page's viewer
  starts from the front, not from behind) and glass left see-through.
"""
import re
import sys
from pathlib import Path

# Pixal3D's GLB ending, in app.py and inference.py alike: a rotation that
# leaves the front on -Z, then the export. glb_export.export() replaces both.
ROTATION_RE = re.compile(
    r"(?:# Apply rotation\n\s*)?rot = np\.array\(\[\s*\[-1,\s*0,\s*0,\s*0\],\s*\[ ?0,\s*0,\s*-1,\s*0\],\s*"
    r"\[ ?0,\s*-1,\s*0,\s*0\],\s*\[ ?0,\s*0,\s*0,\s*1\],?\s*\], dtype=np\.float64\)\s*"
    r"glb\.apply_transform\(rot\)\n")


def use_glb_export(src: str, export_line: str, new_line: str) -> str:
    """Swap Pixal3D's rotation and export for glb_export.export(); fail the
    build if upstream changed either, rather than ship the old export."""
    src, turned = ROTATION_RE.subn("", src)
    assert turned == 1 and src.count(export_line) == 1, "Pixal3D's GLB export changed"
    return src.replace(export_line, new_line).replace(
        "import numpy as np\n", "import numpy as np\nimport glb_export\n", 1)


DINOV3_MIRROR = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
DINOV3 = "facebook/dinov3-vitl16-pretrain-lvd1689m"

path = Path("/workspace/Pixal3D/app.py")
src = path.read_text().replace(DINOV3_MIRROR, DINOV3)

# 0. Replace `import spaces` with a no-op shim so @spaces.GPU(duration=...)
#    decorators become identity functions (HF ZeroGPU is not present on Spark).
#    Also disarm Gradio's safehttpx SSRF guard: the custom index.html posts the
#    uploaded image back as an absolute URL pointing at the server's own LAN IP
#    (e.g. http://192.168.3.16:12158/...). safehttpx rejects RFC1918 hosts and
#    the upload fails before Start Generation can run.
src = src.replace(
    "import spaces",
    (
        "class _FakeSpaces:\n"
        "    @staticmethod\n"
        "    def GPU(*a, **k):\n"
        "        def _d(fn): return fn\n"
        "        return _d\n"
        "spaces = _FakeSpaces()\n"
        "import safehttpx as _shx, socket as _sock\n"
        "async def _shx_allow_lan(hostname, *a, **k):\n"
        "    return _sock.gethostbyname(hostname)\n"
        "_shx.async_validate_url = _shx_allow_lan"
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
    'open("/tmp/pixal3d_ready", "w").close()\n'
    '    app.launch(show_error=True, share=False, '
    'allowed_paths=["/workspace/Pixal3D/tmp", "/workspace/Pixal3D/assets", "/tmp"])',
)

# 4. Serve the page's static files at /px-assets (see index.html below) and
#    the results under /tmp from plain routes. Mounted StaticFiles answer 404
#    behind the Hub: the Hub strips /run/pixal3d-spark before forwarding, while
#    GRADIO_ROOT_PATH makes Starlette look for that prefix in every path.
MOUNTS = ('app.mount("/assets", StaticFiles(directory="assets"), name="assets")\n'
          'app.mount("/tmp", StaticFiles(directory=TMP_DIR), name="tmp")\n')
assert src.count(MOUNTS) == 1, "app.py's static mounts changed"
src = src.replace(MOUNTS, """from fastapi import HTTPException as _HTTPException
from fastapi.responses import FileResponse as _FileResponse

def _serve_folder(prefix, folder):
    base = os.path.realpath(folder)
    @app.get(prefix + "/{path:path}", include_in_schema=False)
    async def _serve(path: str):
        full = os.path.realpath(os.path.join(base, path))
        if not full.startswith(base + os.sep) or not os.path.isfile(full):
            raise _HTTPException(404)
        return _FileResponse(full)

_serve_folder("/px-assets", "assets")
_serve_folder("/tmp", TMP_DIR)
""")

# 5. GLBs through glb_export.py (see its docstring).
src = use_glb_export(src, "glb.export(out_glb, extension_webp=True)", "glb_export.export(glb, out_glb)")

# 6. Rewrite every CDN URL in index.html to point at the locally-vendored copy
#    under /assets/vendor/ so launch is fully offline. If any CDN import fails,
#    the entire <script type="module"> aborts and the UI breaks silently.
ihtml = Path("/workspace/Pixal3D/index.html")
if ihtml.exists():
    h = ihtml.read_text()
    # crypto.randomUUID() requires a secure context (HTTPS or localhost). The
    # Hub is served over plain HTTP on a LAN IP, so this call throws TypeError
    # and the whole <script type="module"> aborts — empty gallery, no upload,
    # no icons. Inject a polyfill that uses crypto.getRandomValues when
    # available and Math.random otherwise.
    polyfill = (
        "if (!crypto.randomUUID) {\n"
        "  crypto.randomUUID = function () {\n"
        "    const b = new Uint8Array(16);\n"
        "    (crypto.getRandomValues ? crypto.getRandomValues(b) : b.forEach((_,i,a)=>a[i]=Math.floor(Math.random()*256)));\n"
        "    b[6] = (b[6] & 0x0f) | 0x40; b[8] = (b[8] & 0x3f) | 0x80;\n"
        "    const h = [...b].map(x => x.toString(16).padStart(2,'0'));\n"
        "    return `${h.slice(0,4).join('')}-${h.slice(4,6).join('')}-${h.slice(6,8).join('')}-${h.slice(8,10).join('')}-${h.slice(10).join('')}`;\n"
        "  };\n"
        "}\n"
    )
    h = h.replace(
        "const sessionId = crypto.randomUUID();",
        polyfill + "        const sessionId = crypto.randomUUID();",
    )
    # The models now face +Z (glb_export), so the viewer starts from the
    # front instead of from behind.
    assert h.count("-180deg 90deg auto") == 2, "Pixal3D's viewer camera changed"
    h = h.replace("-180deg 90deg auto", "0deg 90deg auto")
    for cdn, local in {
        'https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js':
            './px-assets/vendor/gradio-client.min.js',
        'https://unpkg.com/lucide@latest':
            './px-assets/vendor/lucide.min.js',
        'https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js':
            './px-assets/vendor/model-viewer.min.js',
        # Replace the entire Google Fonts <link> with our vendored CSS. Match
        # the full URL so preconnect lines remain (browser just ignores them
        # offline). The href= we replace is the only one that actually loads.
        'https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap':
            './px-assets/vendor/fonts/fonts.css',
    }.items():
        h = h.replace(cdn, local)
    # Every URL relative to the page, so it works at the root and behind the
    # Spark AI Hub's /run/pixal3d-spark/ alike, where a root path would reach
    # the Hub instead. The static files move to /px-assets: Gradio's own
    # /assets route is matched first and answers 404 for them.
    for old, new, n in (
        ("Client.connect(window.location.origin)",
         "Client.connect(new URL('.', window.location.href).href.replace(/\\/$/, ''))", 1),
        ("fetch(`/progress?", "fetch(`progress?", 1),
        ("`/tmp/${filename}`", "`tmp/${filename}`", 2),
        ("'assets/images/", "'./px-assets/images/", None),
        # Hand the cut-out back as the file the server returned, not as its
        # URL: given a URL, the server downloads it, and from inside the
        # container the address the browser used (the Hub) is unreachable or
        # signed-in only.
        ("preprocessedFile = processedUrl;", "preprocessedFile = result.data[0];", 1),
        ("preprocessedFile ? handle_file(preprocessedFile)", "preprocessedFile ? preprocessedFile", 1),
    ):
        assert n is None and old in h or h.count(old) == n, f"index.html changed: {old}"
        h = h.replace(old, new)
    assert "\"/assets/" not in h and "'/assets/" not in h, "index.html has another /assets path"
    ihtml.write_text(h)

path.write_text(src)

# Also patch inference.py for CLI use, same ATTN_BACKEND and DINOv3 fixes.
inf = Path("/workspace/Pixal3D/inference.py")
if inf.exists():
    s = inf.read_text().replace(
        'os.environ["ATTN_BACKEND"] = "flash_attn_3"',
        'os.environ["ATTN_BACKEND"] = "flash_attn"',
    ).replace(DINOV3_MIRROR, DINOV3)
    s = use_glb_export(s, "glb.export(output_path, extension_webp=True)", "glb_export.export(glb, output_path)")
    inf.write_text(s)

print("Patches applied.")
