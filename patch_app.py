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
    'open("/tmp/pixal3d_ready", "w").close()\n    app.launch(show_error=True, share=False)',
)

# 4. Rewrite every CDN URL in index.html to point at the locally-vendored copy
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
    for cdn, local in {
        'https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js':
            '/assets/vendor/gradio-client.min.js',
        'https://unpkg.com/lucide@latest':
            '/assets/vendor/lucide.min.js',
        'https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js':
            '/assets/vendor/model-viewer.min.js',
        # Replace the entire Google Fonts <link> with our vendored CSS. Match
        # the full URL so preconnect lines remain (browser just ignores them
        # offline). The href= we replace is the only one that actually loads.
        'https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap':
            '/assets/vendor/fonts/fonts.css',
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
