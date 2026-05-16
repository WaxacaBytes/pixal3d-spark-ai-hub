# Pixal3D for DGX Spark (aarch64 + CUDA 13 + Blackwell sm_120)
#
# Builds on top of the trellis2-spark base image, which already provides:
#   - CUDA 12.9 + PyTorch nightly (cu129)
#   - flash-attn, nvdiffrast, nvdiffrec, CuMesh, FlexGEMM, o-voxel (compiled for sm_120)
#   - torchvision built from source for Blackwell
#   - transformers, gradio, kornia, timm, trimesh, etc.
#
# Pixal3D adds: diffusers, accelerate, einops, plyfile, MoGe, natten.
# Upstream Pixal3D wheel URLs are x86_64 only — we skip them and rely on the
# source-built equivalents in the base image (or build natten from source).

FROM abelpc/trellis2-spark:v2.0

ARG DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda-12.9
ENV TORCH_CUDA_ARCH_LIST="12.1+PTX"

# Clone Pixal3D into its own workspace
ARG PIXAL3D_REPO=https://github.com/TencentARC/Pixal3D.git
ARG PIXAL3D_REF=master
RUN git clone -b "$PIXAL3D_REF" "$PIXAL3D_REPO" /workspace/Pixal3D

# Install the pure-Python deps Pixal3D adds on top of the trellis2 base.
# Anything already present in the base (torch, gradio, transformers, kornia,
# timm, trimesh, opencv, etc.) is intentionally NOT re-pinned to avoid breaking
# the carefully compiled CUDA stack.
RUN pip install --no-cache-dir \
    diffusers==0.37.1 \
    accelerate==1.13.0 \
    einops==0.8.2 \
    plyfile==1.1.3 \
    easydict==1.13 \
    zstandard==0.25.0 \
    "gradio>=6.14,<7"

# MoGe-2 (monocular geometry estimator) — pure Python, pulls from git
RUN pip install --no-cache-dir git+https://github.com/microsoft/MoGe.git

# natten (Neighborhood Attention) — prebuilt wheel for aarch64 + CUDA 12.9 +
# Python 3.12, compiled natively on DGX Spark and hosted as a release asset on
# this repo. This avoids ~2hr QEMU-emulated source builds in GitHub Actions.
RUN pip install --no-cache-dir \
    https://github.com/WaxacaBytes/pixal3d-spark-ai-hub/releases/download/wheels-v1/NATTEN-0.21.6-cp312-cp312-linux_aarch64.whl

# Bake all UI JS dependencies locally so the page does not need internet at
# load time. Without this, the entire app.py <script type="module"> aborts
# when any CDN import fails — empty gallery, broken image picks, no icons.
RUN mkdir -p /workspace/Pixal3D/assets/vendor /workspace/Pixal3D/assets/vendor/fonts && \
    curl -fsSL https://cdn.jsdelivr.net/npm/@gradio/client@2.2.0/dist/index.min.js \
      -o /workspace/Pixal3D/assets/vendor/gradio-client.min.js && \
    curl -fsSL https://unpkg.com/lucide@latest \
      -o /workspace/Pixal3D/assets/vendor/lucide.min.js && \
    curl -fsSL https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js \
      -o /workspace/Pixal3D/assets/vendor/model-viewer.min.js && \
    curl -fsSL -A "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 Chrome/120 Safari/537.36" \
      "https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap" \
      -o /workspace/Pixal3D/assets/vendor/fonts/fonts.css && \
    cd /workspace/Pixal3D/assets/vendor/fonts && \
    for u in $(grep -oE 'https://fonts.gstatic.com/[^)"]+\.woff2?' fonts.css | sort -u); do \
        f=$(basename "$u"); \
        curl -fsSL "$u" -o "$f"; \
        sed -i "s|$u|./$f|g" fonts.css; \
    done

# Patch app.py: skip x86_64 utils3d wheel reinstall, force flash_attn v2
# (no flash_attn_3 on aarch64), disable Gradio share tunneling, drop the
# init-complete marker, and rewrite the index.html CDN import.
COPY patch_app.py /tmp/patch_app.py
RUN python /tmp/patch_app.py && rm /tmp/patch_app.py

WORKDIR /workspace/Pixal3D

# Pixal3D ships a modified copy of the trellis2 package as ./pixal3d/ (adds
# Pixal3DImageTo3DPipeline, tweaks trellis2_image_to_3d, etc.). app.py imports
# it as `trellis2`, so we symlink and put the Pixal3D cwd on PYTHONPATH ahead
# of the base image's TRELLIS.2 to make the Pixal3D version win.
RUN ln -s /workspace/Pixal3D/pixal3d /workspace/Pixal3D/trellis2
ENV PYTHONPATH="/workspace/Pixal3D:${PYTHONPATH}"

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
ENV HF_HOME="/workspace/cache/huggingface"
ENV TRITON_CACHE_DIR="/workspace/cache/triton"
ENV TORCH_HOME="/workspace/cache/torch"
ENV ATTN_BACKEND="flash-attn"

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
