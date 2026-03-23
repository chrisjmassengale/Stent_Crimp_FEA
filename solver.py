"""
solver.py — Stent crimping and deployment solver.

CRIMPING: Pure geometric radial scale. Z frozen. No FEA.
DEPLOYMENT: Progressive radial release, top-to-bottom (+Z direction).
  Returns frame metadata so deform.py can apply per-vertex z-based S-curves.
"""

import numpy as np
from typing import List, Tuple, Dict

from topology import BeamNetwork
from utils import smooth_step

RELEASE_FRACTION = 0.80   # front reaches z_min at 80% of deploy steps; last 20% = settle


def simulate(network: BeamNetwork,
             crimp_diameter_mm: float,
             deployed_diameter_mm: float,
             n_crimp_steps: int = 50,
             n_deploy_steps: int = 50,
             verbose: bool = True) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    Returns
    -------
    frames : list of node-position arrays (N_nodes, 3) — one per step
    meta   : parallel list of dicts describing each frame:
               crimp:  {'type':'crimp', 'scale': float, 'crimp_r': float,
                        'deploy_r': float, 'center_xy': ndarray}
               deploy: {'type':'deploy', 'z_front_norm': float, 't_step': float,
                        'crimp_r': float, 'deploy_r': float,
                        'z_min': float, 'z_span': float, 'center_xy': ndarray}
    """
    from tqdm import tqdm

    ref_pos   = network.node_positions.copy()
    node_ids  = network.node_ids
    center_xy = ref_pos[:, :2].mean(axis=0)

    dx = ref_pos[:, 0] - center_xy[0]
    dy = ref_pos[:, 1] - center_xy[1]
    r_natural = float(np.median(np.sqrt(dx**2 + dy**2)))

    crimp_r  = crimp_diameter_mm  / 2.0
    deploy_r = deployed_diameter_mm / 2.0

    if verbose:
        print(f"[solver] Natural radius = {r_natural:.2f} mm  |  "
              f"crimp_r = {crimp_r:.2f} mm  |  deploy_r = {deploy_r:.2f} mm")

    # ── Phase 1: Pure geometric crimping ──────────────────────────────────────
    if verbose:
        print(f"[solver] Phase 1: Crimping ({n_crimp_steps} steps) ...")

    crimp_frames, crimp_meta = [], []

    for step in tqdm(range(n_crimp_steps), desc="Crimping", disable=not verbose):
        t      = smooth_step((step + 1) / n_crimp_steps)
        target = r_natural + t * (crimp_r - r_natural)
        scale  = target / r_natural

        pos = ref_pos.copy()
        pos[:, 0] = center_xy[0] + (ref_pos[:, 0] - center_xy[0]) * scale
        pos[:, 1] = center_xy[1] + (ref_pos[:, 1] - center_xy[1]) * scale
        pos[:, 2] = ref_pos[:, 2]   # Z frozen

        # Validation: Z must never drift
        assert np.max(np.abs(pos[:, 2] - ref_pos[:, 2])) < 1e-9, "Z drift in crimp!"

        crimp_frames.append(pos)
        crimp_meta.append({
            'type': 'crimp', 'scale': scale,
            'crimp_r': crimp_r, 'deploy_r': deploy_r,
            'center_xy': center_xy,
        })

        if verbose and step in (0, n_crimp_steps // 2, n_crimp_steps - 1):
            r_now = np.sqrt((pos[:, 0] - center_xy[0])**2 +
                            (pos[:, 1] - center_xy[1])**2)
            print(f"  step {step+1:3d}: R mean={r_now.mean():.2f}  "
                  f"target={target:.2f}  Z std={np.std(pos[:,2]-ref_pos[:,2]):.2e}")

    if verbose:
        print("[solver] Crimping validation: PASSED")

    # ── Phase 2: Deployment — tube retracts in +Z direction (top-to-bottom) ──
    # The tube opening travels from z_MAX down to z_MIN.
    # z_norm = (z_max - z) / z_span  →  0 = top (released first), 1 = bottom (last)
    if verbose:
        print(f"[solver] Phase 2: Deployment ({n_deploy_steps} steps, top→bottom) ...")

    z_ref  = ref_pos[:, 2]
    z_min  = float(z_ref.min())
    z_max  = float(z_ref.max())
    z_span = z_max - z_min if z_max > z_min else 1.0

    deploy_frames, deploy_meta = [], []

    for step in tqdm(range(n_deploy_steps), desc="Deployment", disable=not verbose):
        t_step     = (step + 1) / n_deploy_steps
        t_front    = min(t_step / RELEASE_FRACTION, 1.0)

        # Node positions (for reference / strain tracking — actual vertex
        # deformation is done per-vertex in deform.py using z directly)
        z_norm_nodes = (z_max - z_ref) / z_span   # 0=top, 1=bottom

        pos = crimp_frames[-1].copy()
        for n in node_ids:
            zn = z_norm_nodes[n]
            d  = t_front - zn                      # + = released
            # Use same formula as deform.py for consistency
            t_zone  = float(np.clip(d / 0.20 + 0.5, 0, 1))
            r_frac  = t_zone * t_zone * (3 - 2 * t_zone)  # smoothstep
            snap_d  = float(np.clip((d - 0.10) / 0.20, 0, 1))
            r_snap  = 1.0 - (1.0 - snap_d)**3
            r_frac  = max(r_frac, r_snap)
            target  = crimp_r + r_frac * (deploy_r - crimp_r)

            dxn = ref_pos[n, 0] - center_xy[0]
            dyn = ref_pos[n, 1] - center_xy[1]
            r0  = np.sqrt(dxn**2 + dyn**2)
            if r0 > 1e-8:
                s = target / r0
                pos[n, 0] = center_xy[0] + dxn * s
                pos[n, 1] = center_xy[1] + dyn * s
            pos[n, 2] = ref_pos[n, 2]   # Z frozen

        deploy_frames.append(pos)
        deploy_meta.append({
            'type': 'deploy',
            'z_front_norm': t_front,
            't_step': t_step,
            'crimp_r': crimp_r,
            'deploy_r': deploy_r,
            'z_min': z_min,
            'z_span': z_span,
            'center_xy': center_xy,
        })

    all_frames = crimp_frames + deploy_frames
    all_meta   = crimp_meta   + deploy_meta

    if verbose:
        final_r = np.sqrt((deploy_frames[-1][:, 0] - center_xy[0])**2 +
                          (deploy_frames[-1][:, 1] - center_xy[1])**2)
        print(f"[solver] Done — {len(all_frames)} frames")
        print(f"  Crimped  R mean: {np.sqrt((crimp_frames[-1][:,0]-center_xy[0])**2+(crimp_frames[-1][:,1]-center_xy[1])**2).mean():.2f} mm")
        print(f"  Deployed R mean: {final_r.mean():.2f} mm (target {deploy_r:.2f})")

    return all_frames, all_meta
