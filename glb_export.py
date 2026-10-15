"""How Pixal3D's models are written out as GLB files on the Spark.

Pixal3D's own app.py and inference.py end the same way: o_voxel's to_glb,
a rotation, an export. Two things go wrong there, and export() fixes both:

- Pixal3D works Z-up; glTF is Y-up with the front of a model on +Z. Pixal3D's
  rotation adds a half turn that leaves the front on -Z, so viewers open the
  model from behind (its own web page made up for it by starting its camera
  from behind). GLB_ROTATION is the plain Z-up to Y-up turn.
- to_glb keeps the transparency Pixal3D predicts in the alpha of the base
  colour texture but always marks the material OPAQUE, so glass (a visor, a
  snow globe) renders as a solid wall. Faces Pixal3D made see-through get a
  BLEND copy of the material, sharing its textures; the rest stays OPAQUE,
  so it still draws solid, in the right order, in every viewer.

Pixal3D's app.py and inference.py call it (patch_app.py), and so does the
Spark AI Hub's Pixal3D engine (pixal3d-api-spark-ai-hub), which builds on
this image.
"""
import numpy as np
import trimesh

GLB_ROTATION = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
# A face whose corners average below this alpha (of 255) is see-through.
SEE_THROUGH = 250


def with_transparency(mesh):
    """The mesh, or a scene of its solid and its see-through parts."""
    material = mesh.visual.material
    texture = np.asarray(material.baseColorTexture)
    if texture.ndim != 3 or texture.shape[2] != 4:
        return mesh
    h, w = texture.shape[:2]
    uv = mesh.visual.uv
    alpha = texture[np.clip(((1 - uv[:, 1]) * h).astype(int), 0, h - 1),
                    np.clip((uv[:, 0] * w).astype(int), 0, w - 1), 3].astype(np.float32)
    clear = alpha[mesh.faces].mean(axis=1) < SEE_THROUGH
    if not clear.any():
        return mesh
    parts = []
    for faces, mode in ((~clear, "OPAQUE"), (clear, "BLEND")):
        if faces.any():
            part = mesh.submesh([np.flatnonzero(faces)], append=True)
            part.visual.material = material.copy()       # its own mode, the same pictures
            part.visual.material.baseColorTexture = material.baseColorTexture
            part.visual.material.metallicRoughnessTexture = material.metallicRoughnessTexture
            part.visual.material.alphaMode = mode
            parts.append(part)
    return trimesh.Scene(parts)


def export(glb, path) -> None:
    """Write to_glb's mesh to `path`: front on +Z, glass see-through."""
    glb.apply_transform(GLB_ROTATION)
    with_transparency(glb).export(path, extension_webp=True, include_normals=True)
