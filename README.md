# pixal3d-spark-ai-hub

Native ARM64 + CUDA Docker build of [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) for the NVIDIA DGX Spark, distributed as part of [Spark AI Hub](https://github.com/WaxacaBytes/spark-ai-hub).

Image: `abelpc/pixal3d-spark:latest`

Built on top of [`abelpc/trellis2-spark:v2.0`](https://github.com/WaxacaBytes/trellis2-spark) (CUDA 12.9 + PyTorch nightly + native sm_120 CUDA extensions). Adds Pixal3D's extra deps (diffusers, accelerate, einops, plyfile, MoGe, natten) and runs `app_local.py` on port 7860.

Per the [Spark AI Hub manifest](https://github.com/WaxacaBytes/spark-ai-hub/blob/main/MANIFEST.md):

- Offline-first after first launch
- ARM64 + CUDA 13 native — no x86 emulation
- One-button install/launch from Spark AI Hub
