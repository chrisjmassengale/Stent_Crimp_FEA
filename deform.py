"""
deform.py — Pure cylindrical coordinate mesh deformation.

Every vertex moves purely radially — theta is frozen, so no twisting
is geometrically possible.  Every vertex at the same Z gets the same
scale factor — no differential stretching between neighbors.
Polygon count is identical to input — we only touch positions, never topology.

No special cases. No feature detection. No per-feature overrides.
Just cylindrical coordinate scaling applied identically to all vertices.
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from topology import BeamNetwork


# ── defaults ──────────────────────────────────────────────────────────────────
TRANSITION_FRAC  = 0.45   # transition zone = 45% of stent height
SNAP_SPEED       = 3.0    # exponent in snap curve: 1 - (1-t)^n
CROWN_DWELL      = 0.60   # fraction of crown arm that must be free before release
EXPANSION_EXP    = 0.6    # global expansion curve exponent


# ── helpers ───────────────────────────────────────────────────────────────────

def _smoothstep(x):
    """Hermite smoothstep: clamp(x,0,1)^2 * (3 - 2*clamp(x,0,1))."""
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _snap_curve(t, speed=SNAP_SPEED):
    """1 - (1-t)^speed — fast initial expansion, smooth settle."""
    return 1.0 - (1.0 - t) ** speed


# ── frame export (the only public entry point) ────────────────────────────────

def export_frames(mesh: trimesh.Trimesh,
                  network: BeamNetwork,
                  frames: List[np.ndarray],
                  meta: List[Dict],
                  output_dir: str,
                  verbose: bool = True,
                  transition_frac: float = TRANSITION_FRAC,
                  snap_speed: float = SNAP_SPEED,
                  crown_dwell: float = CROWN_DWELL,
                  expansion_exponent: float = EXPANSION_EXP,
                  **kwargs) -> List[str]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Store original vertex positions (never modified) ──────────────────────
    orig = mesh.vertices.copy()
    n_verts = len(orig)
    n_faces = len(mesh.faces)

    # Stent axis center (from solver — consistent across all frames)
    cxy = meta[0]['center_xy']
    cx, cy = float(cxy[0]), float(cxy[1])

    # ── Precompute cylindrical coords once ────────────────────────────────────
    dx = orig[:, 0] - cx
    dy = orig[:, 1] - cy
    r_orig = np.sqrt(dx ** 2 + dy ** 2)
    theta  = np.arctan2(dy, dx)
    z_orig = orig[:, 2]
    r_center = np.median(r_orig)               # strut centerline radius
    r_offset = r_orig - r_center               # cross-section offset (preserved)

    # ── Crown geometry for dwell (only thing that varies by Z) ────────────────
    node_z = network.node_positions[:, 2]
    z_span_mesh = float(z_orig.max() - z_orig.min())

    cluster_tol = 0.05 * z_span_mesh
    z_sorted = np.sort(node_z)
    cluster_centers = [z_sorted[0]]
    for zv in z_sorted[1:]:
        if zv - cluster_centers[-1] > cluster_tol:
            cluster_centers.append(zv)
        else:
            cluster_centers[-1] = 0.5 * (cluster_centers[-1] + zv)
    n_z_levels = len(cluster_centers)
    n_cells = max(1, (n_z_levels - 1) // 2)
    crown_arm_length = z_span_mesh / (2.0 * n_cells)
    cell_height = 2.0 * crown_arm_length

    # Graduated crown dwell
    z_in_cell = (z_orig - float(z_orig.min())) % cell_height
    z_cell_frac = z_in_cell / cell_height
    crown_proximity = 2.0 * np.abs(z_cell_frac - 0.5)
    dwell_per_vertex = crown_arm_length * crown_dwell * crown_proximity

    if verbose:
        print(f"[deform] Input: {n_verts} verts, {n_faces} faces")
        print(f"[deform] Center: ({cx:.2f}, {cy:.2f})  "
              f"R range: [{r_orig.min():.2f}, {r_orig.max():.2f}] mm")
        print(f"[deform] Crown: {n_cells} cells, arm={crown_arm_length:.2f} mm, "
              f"max_dwell={crown_arm_length * crown_dwell:.2f} mm")
        print(f"[deform] Transition={transition_frac:.2f}  snap={snap_speed:.1f}  "
              f"dwell={crown_dwell:.2f}  expansion_exp={expansion_exponent:.2f}")

    # ── Per-frame deformation ─────────────────────────────────────────────────
    paths: List[str] = []

    for idx, fm in tqdm(enumerate(meta), total=len(meta),
                        desc="Exporting STL frames", disable=not verbose):

        crimp_r  = fm['crimp_r']
        deploy_r = fm['deploy_r']

        # ── CRIMP: uniform radial scale ───────────────────────────────────────
        if fm['type'] == 'crimp':
            r_new = r_center * fm['scale'] + r_offset
            z_new = z_orig

        # ── DEPLOY: z-based tube-retraction with snap-back ────────────────────
        else:
            z_min  = fm['z_min']
            z_span = fm['z_span']
            z_max  = z_min + z_span
            z_front = fm['z_front_norm']

            tube_tip_z = z_max - z_front * z_span
            trans_len  = transition_frac * z_span

            # Crown dwell
            z_effective = z_orig - dwell_per_vertex

            # Per-vertex released fraction (smoothstep over transition zone)
            d = z_effective - tube_tip_z
            released = _smoothstep(d / trans_len)
            snap     = _snap_curve(released, snap_speed)

            # Two-phase superelastic deployment
            # Phase 1 — snap-back : strut exits tube → springs to 97 % of
            #            deployed radius (superelastic, instantaneous per-vertex)
            # Phase 2 — thermal   : slow creep from 97 % → 100 % as struts
            #            warm above Af into body-temperature environment.
            #            expansion_exponent controls warm-up rate (higher = faster)
            _SNAP_FRAC   = 0.97
            t_global     = float(np.clip(z_front, 0., 1.))   # = released length frac
            t_release_v  = np.clip((z_max - z_effective) / max(z_span, 1e-6), 0., 1.)
            t_out_v      = np.clip(t_global - t_release_v, 0., 1.)
            thermal_v    = _smoothstep(np.minimum(t_out_v * expansion_exponent * 5., 1.))

            r_snap_tgt   = crimp_r + _SNAP_FRAC * (deploy_r - crimp_r)
            r_centerline = (crimp_r
                            + (r_snap_tgt - crimp_r)   * snap
                            + (deploy_r   - r_snap_tgt) * thermal_v * released)
            r_new = r_centerline + r_offset
            z_new = z_orig

        # ── Convert back to cartesian — SAME FOR EVERY VERTEX ────────────────
        new_verts = np.empty_like(orig)
        new_verts[:, 0] = cx + r_new * np.cos(theta)
        new_verts[:, 1] = cy + r_new * np.sin(theta)
        new_verts[:, 2] = z_new

        # ── Write STL ─────────────────────────────────────────────────────────
        deformed = trimesh.Trimesh(
            vertices=new_verts,
            faces=mesh.faces.copy(),
            process=False,
        )
        fname = out_path / f"frame_{idx:03d}.stl"
        deformed.export(str(fname))
        paths.append(str(fname))

    if verbose:
        print(f"[deform] Wrote {len(paths)} frames ({n_verts}v {n_faces}f each)")

    return paths


# ── diagnostics (kept for simulate.py compatibility) ──────────────────────────

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
