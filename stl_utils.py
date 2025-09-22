import numpy as np
import open3d as o3d

def load_stl_with_open3d(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load STL file with Open3D and return vertices, triangles, and normals."""
    mesh = o3d.io.read_triangle_mesh(filepath)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return (
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.triangles, dtype=np.int32),
        np.asarray(mesh.triangle_normals, dtype=np.float32),
    )
