"""
deform.py — Mesh deformation and STL frame export.

Strategy: delta_r preservation.
  - Precompute delta_r = R_vertex - R_nearest_node (radial cross-section offset).
  - Crimp frames: new_R = IDW(R_node_new) + delta_r
  - Deploy frames: per-vertex z-based S-curve gives target_r_center,
    then new_R = target_r_center + delta_r
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import List, Dict
from scipy.spatial import KDTree
from tqdm import tqdm

from topology import BeamNetwork


# ── constants ─────────────────────────────────────────────────────────────────
TRANSITION_CELLS = 1.5   # transition zone = 1.5 × cell_height (trumpet bell flare)
FORESHORTEN_MAX  = 0.08  # max 8% axial compression at full deployment
_IDW_K      = 6
_IDW_POWER  = 3.0


# ── pre-computation ────────────────────────────────────────────────────────────

def _precompute(vertices: np.ndarray,
                ref_nodes: np.ndarray,
                center_xy: np.ndarray) -> dict:
    """
    Precompute per-vertex data that is fixed for the entire animation.

    Returns a dict with:
      idxs        (V, K)  — KD-tree nearest-node indices
      weights     (V, K)  — IDW weights
      nn          (V,)    — single nearest-node index
      R_ref_nodes (N,)    — reference radial position of each node
      delta_r     (V,)    — radial offset of vertex from nearest node
      theta_v     (V,)    — angular position of vertex around stent axis
      z_norm_v    (V,)    — normalised axial position (0=top, 1=bottom)
      z_ref_nodes (N,)    — reference Z of each node
      center_xy   (2,)
    """
    cx, cy = center_xy
    K = min(_IDW_K, len(ref_nodes))

    # Node radial positions
    dx_n = ref_nodes[:, 0] - cx
    dy_n = ref_nodes[:, 1] - cy
    R_ref_nodes = np.sqrt(dx_n**2 + dy_n**2)

    # KDTree nearest-node lookup
    tree = KDTree(ref_nodes)
    dists, idxs = tree.query(vertices, k=K)

    # IDW weights
    w = 1.0 / (dists + 1e-8) ** _IDW_POWER
    w /= w.sum(axis=1, keepdims=True)

    nn = idxs[:, 0]  # nearest node per vertex

    # Vertex polar coordinates
    vx = vertices[:, 0] - cx
    vy = vertices[:, 1] - cy
    R_v = np.sqrt(vx**2 + vy**2)
    theta_v = np.arctan2(vy, vx)

    # delta_r: radial offset from nearest node (preserves cross-section thickness)
    delta_r = R_v - R_ref_nodes[nn]

    # Normalised axial position per vertex (weighted from nearest nodes)
    z_ref = ref_nodes[:, 2]
    z_min = float(z_ref.min())
    z_max = float(z_ref.max())
    z_span = z_max - z_min if z_max > z_min else 1.0
    z_norm_nodes = (z_max - z_ref) / z_span   # 0=top, 1=bottom
    z_norm_v = (w * z_norm_nodes[idxs]).sum(axis=1)

    return {
        'idxs':         idxs,
        'weights':      w,
        'nn':           nn,
        'R_ref_nodes':  R_ref_nodes,
        'z_ref_nodes':  z_ref.copy(),
        'delta_r':      delta_r,
        'theta_v':      theta_v,
        'z_norm_v':     z_norm_v,
        'center_xy':    center_xy,
    }


# ── per-frame deformation ─────────────────────────────────────────────────────

def _apply_radial(vertices, R_new_v, theta_v, zdelta_v, cx, cy):
    """Apply new radial distances and Z deltas to produce new vertex array."""
    R_new_v = np.maximum(R_new_v, 0.0)
    out = vertices.copy()
    mask = R_new_v > 1e-8
    out[mask, 0] = cx + R_new_v[mask] * np.cos(theta_v[mask])
    out[mask, 1] = cy + R_new_v[mask] * np.sin(theta_v[mask])
    out[:, 2] = vertices[:, 2] + zdelta_v
    return out


def deform_crimp(vertices: np.ndarray,
                 new_nodes: np.ndarray,
                 pre: dict) -> np.ndarray:
    """
    Crimp frame: R_vertex = IDW(R_node_new) + delta_r.
    Preserves strut cross-section radial offset throughout crimping.
    """
    cx, cy = pre['center_xy']
    w, idxs = pre['weights'], pre['idxs']

    dx_new = new_nodes[:, 0] - cx
    dy_new = new_nodes[:, 1] - cy
    R_new_nodes = np.sqrt(dx_new**2 + dy_new**2)

    R_center_v = (w * R_new_nodes[idxs]).sum(axis=1)
    R_new_v = R_center_v + pre['delta_r']

    # Z: IDW of node Z displacement (solver freezes Z, so this is ~0)
    zdelta_nodes = new_nodes[:, 2] - pre['z_ref_nodes']
    zdelta_v = (w * zdelta_nodes[idxs]).sum(axis=1)

    return _apply_radial(vertices, R_new_v, pre['theta_v'], zdelta_v, cx, cy)


def deform_deploy(vertices: np.ndarray,
                  new_nodes: np.ndarray,
                  meta: dict,
                  pre: dict) -> np.ndarray:
    """
    Deploy frame: per-vertex z-based smooth trumpet-bell transition.

    Uses absolute cell-height-based transition length (matching real stent photos):
      r(z) = r_crimp + (r_deploy - r_crimp) * smoothstep((z - z_tube_tip) / transition_length)
    where transition_length = 1.5 * cell_height.

    Also applies gentle axial foreshortening toward stent midpoint:
      z_scale = 1.0 - 0.08 * r_frac
    """
    cx, cy   = pre['center_xy']
    w, idxs  = pre['weights'], pre['idxs']
    z_front  = meta['z_front_norm']
    crimp_r  = meta['crimp_r']
    deploy_r = meta['deploy_r']
    z_span   = meta['z_span']
    z_min    = meta['z_min']

    # Transition window in normalised z, based on cell_height
    cell_height = meta.get('cell_height', z_span / 6.0)
    trans_norm  = TRANSITION_CELLS * cell_height / z_span

    # d = how far past the tube tip this vertex is (positive = released)
    d = z_front - pre['z_norm_v']

    # Smooth trumpet-bell transition: smoothstep over transition_length
    t_raw  = np.clip(d / trans_norm, 0.0, 1.0)
    r_frac = t_raw * t_raw * (3.0 - 2.0 * t_raw)   # smoothstep

    target_r_center = crimp_r + r_frac * (deploy_r - crimp_r)
    R_new_v = target_r_center + pre['delta_r']

    # Axial foreshortening: compress toward stent midpoint proportional to deployment
    z_mid_norm = 0.5
    z_norm_v   = pre['z_norm_v']
    z_scale    = 1.0 - FORESHORTEN_MAX * r_frac
    # Convert foreshortening to absolute z delta
    # z_norm=0 is top (z_max), z_norm=1 is bottom (z_min)
    # Compress toward midpoint in normalised coords, then convert to mm
    z_foreshorten = (z_norm_v - z_mid_norm) * (1.0 - z_scale) * z_span

    zdelta_nodes = new_nodes[:, 2] - pre['z_ref_nodes']
    zdelta_v = (w * zdelta_nodes[idxs]).sum(axis=1)
    # foreshortening adds a z shift (positive z_foreshorten means move toward mid)
    # z_norm > 0.5 (bottom half) → shift up (+Z), z_norm < 0.5 (top half) → shift down (-Z)
    zdelta_v = zdelta_v + z_foreshorten

    return _apply_radial(vertices, R_new_v, pre['theta_v'], zdelta_v, cx, cy)


# ── frame export ───────────────────────────────────────────────────────────────

def export_frames(mesh: trimesh.Trimesh,
                  network: BeamNetwork,
                  frames: List[np.ndarray],
                  meta: List[Dict],
                  output_dir: str,
                  verbose: bool = True) -> List[str]:
    """
    Export one STL file per simulation frame.

    Parameters
    ----------
    mesh       : original undeformed mesh
    network    : beam network (reference node positions)
    frames     : list of (N_nodes, 3) node-position arrays  (from solver)
    meta       : parallel list of frame metadata dicts      (from solver)
    output_dir : directory to write STL files
    verbose    : show progress bar
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ref_nodes = network.node_positions
    center_xy = ref_nodes[:, :2].mean(axis=0)

    if verbose:
        print("[deform] Pre-computing vertex assignments...")
    pre = _precompute(mesh.vertices, ref_nodes, center_xy)

    n_frames = len(frames)
    paths = []

    iter_ = tqdm(enumerate(zip(frames, meta)), total=n_frames,
                 desc="Exporting STL frames", disable=not verbose)

    for idx, (node_pos, frame_meta) in iter_:
        if frame_meta.get('type') == 'deploy':
            new_verts = deform_deploy(mesh.vertices, node_pos, frame_meta, pre)
        else:
            new_verts = deform_crimp(mesh.vertices, node_pos, pre)

        deformed = trimesh.Trimesh(
            vertices=new_verts,
            faces=mesh.faces.copy(),
            process=False
        )
        fname = out_path / f"frame_{idx:03d}.stl"
        deformed.export(str(fname))
        paths.append(str(fname))

    if verbose:
        print(f"[deform] Wrote {n_frames} STL frames to {output_dir!r}")

    return paths


# ── diagnostics ────────────────────────────────────────────────────────────────

def check_mesh_quality(mesh: trimesh.Trimesh) -> dict:
    """Return a dict of mesh quality metrics."""
    return {
        "vertices":     len(mesh.vertices),
        "faces":        len(mesh.faces),
        "watertight":   mesh.is_watertight,
        "volume":       float(mesh.volume) if mesh.is_watertight else None,
        "bounds_min":   mesh.bounds[0].tolist(),
        "bounds_max":   mesh.bounds[1].tolist(),
        "euler_number": mesh.euler_number,
    }


def validate_deformation(ref_mesh: trimesh.Trimesh,
                         def_mesh: trimesh.Trimesh,
                         max_stretch_ratio: float = 5.0) -> bool:
    """Check that the deformation is physically reasonable."""
    ref_edges = ref_mesh.edges_unique
    ref_lens = np.linalg.norm(
        ref_mesh.vertices[ref_edges[:, 0]] - ref_mesh.vertices[ref_edges[:, 1]], axis=1
    )
    def_lens = np.linalg.norm(
        def_mesh.vertices[ref_edges[:, 0]] - def_mesh.vertices[ref_edges[:, 1]], axis=1
    )
    ratios = def_lens / (ref_lens + 1e-12)
    return not (ratios.max() > max_stretch_ratio or ratios.min() < 1.0 / max_stretch_ratio)
