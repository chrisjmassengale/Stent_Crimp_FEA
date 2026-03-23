"""
Generate a simple parametric stent STL for pipeline testing.
Creates a zigzag-crown stent mesh (open shell) similar to a real laser-cut Nitinol stent.
"""

import numpy as np
import trimesh
from trimesh.creation import cylinder

# Stent parameters
R      = 14.0    # mm — deployed radius
L_stent = 30.0   # mm — stent length
N_circ  = 8      # struts per ring (crown count)
N_rings = 5      # number of crown rings
strut_r = 0.2    # strut cross-section radius
n_seg   = 8      # segments per strut cylinder

meshes = []

def add_strut(p1, p2):
    """Add a cylinder strut from p1 to p2."""
    d = p2 - p1
    L = np.linalg.norm(d)
    if L < 1e-6:
        return
    c = cylinder(radius=strut_r, height=L, sections=n_seg)
    # Default cylinder is along Z; transform to align with d
    from utils import rotation_matrix_to_align
    axis = d / L
    R3 = rotation_matrix_to_align(np.array([0., 0., 1.]), axis)
    T = np.eye(4)
    T[:3, :3] = R3
    T[:3, 3]  = (p1 + p2) / 2
    c.apply_transform(T)
    meshes.append(c)

# Build crown rings
for ir in range(N_rings):
    z0 = L_stent * ir / (N_rings - 1)
    z1 = L_stent * (ir + 0.5) / (N_rings - 1) if ir < N_rings - 1 else None

    for ic in range(N_circ):
        theta0 = 2 * np.pi * ic / N_circ
        theta1 = 2 * np.pi * ((ic + 1) % N_circ) / N_circ
        theta_mid = (theta0 + theta1) / 2

        p_low_l  = np.array([R * np.cos(theta0), R * np.sin(theta0), z0])
        p_low_r  = np.array([R * np.cos(theta1), R * np.sin(theta1), z0])

        # Up-stroke of crown
        if z1 is not None:
            z_top = L_stent * (ir + 0.5) / (N_rings - 1)
            p_top = np.array([R * np.cos(theta_mid), R * np.sin(theta_mid), z_top])
            add_strut(p_low_l, p_top)
            add_strut(p_low_r, p_top)

        # Down-stroke connecting to next ring
        if ir > 0:
            z_prev_top = L_stent * (ir - 0.5) / (N_rings - 1)
            p_prev_top = np.array([R * np.cos(theta_mid), R * np.sin(theta_mid), z_prev_top])
            add_strut(p_prev_top, p_low_l)
            add_strut(p_prev_top, p_low_r)

# Merge all strut cylinders
print(f"Combining {len(meshes)} strut meshes...")
stent = trimesh.util.concatenate(meshes)
stent = trimesh.Trimesh(vertices=stent.vertices, faces=stent.faces, process=True)
print(f"Final mesh: {len(stent.vertices)} vertices, {len(stent.faces)} faces")

out_path = "test_stent.stl"
stent.export(out_path)
print(f"Saved to {out_path}")
