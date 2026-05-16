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

# natten (Neighborhood Attention). The wheels Pixal3D points at are x86_64,
# so we build from source for aarch64/sm_120. NATTEN_CUDA_ARCH controls the
# CUDA arch list; MAX_JOBS keeps memory in check on GB10.
RUN MAX_JOBS=2 NATTEN_CUDA_ARCH="12.1" \
    pip install --no-cache-dir --no-build-isolation natten==0.21.6 \
    || pip install --no-cache-dir --no-build-isolation natten

# Patch app.py: skip x86_64 utils3d wheel reinstall, force flash_attn v2
# (no flash_attn_3 on aarch64), disable Gradio share tunneling (offline-first).
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
