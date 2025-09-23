import numpy as np
import open3d as o3d

def load_stl_file(filepath: str):
    """Load STL into numpy arrays and print info. Returns (verts, tris, norms)."""
    mesh = o3d.io.read_triangle_mesh(filepath)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    tris  = np.asarray(mesh.triangles, dtype=np.int32)
    norms = np.asarray(mesh.triangle_normals, dtype=np.float32)

    # Compute bounding box for debug/info
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    center = (vmin + vmax) / 2
    size = vmax - vmin

       # Round for neat printing
    vmin = np.round(vmin, 3)
    vmax = np.round(vmax, 3)
    center = np.round(center, 3)
    size = np.round(size, 3)

    print(f"[STL] Loaded file: {filepath}")
    print(f"[STL] Vertices: {len(verts)} | Triangles: {len(tris)}")
    print(f"[STL] Bounding box min: {vmin}, max: {vmax}")
    print(f"[STL] Size: {size}")
    print(f"[STL] Center: {center}")

    return verts, tris, norms

