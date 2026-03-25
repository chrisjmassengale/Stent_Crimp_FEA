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


# ══════════════════════════════════════════════════════════════════════════════
# Cross-section frame preservation  (Step 1)
#
# For every mesh vertex we store THREE scalar offsets in the strut's LOCAL FRAME:
#   t_comp  — along the strut centreline direction
#   n_comp  — perpendicular to strut, in the cylinder tangent plane  (WIDTH)
#   r_comp  — perpendicular to strut, radially outward               (THICKNESS)
#
# Because the frame rotates with the strut, these offsets are invariant —
# i.e. the cross-section is RIGID regardless of how the centreline deforms.
# ══════════════════════════════════════════════════════════════════════════════

def _local_frame_at(cp, t_hat, cx, cy):
    """
    Build the orthonormal local frame at one centreline point.

    Parameters
    ----------
    cp    : (3,) centreline point
    t_hat : (3,) unit strut-tangent vector
    cx,cy : stent axis centre

    Returns
    -------
    r_hat : (3,) thickness direction  (perp to strut, radially outward component)
    n_hat : (3,) width direction      (perp to strut, in cylinder tangent plane)
    """
    dr     = np.array([cp[0] - cx, cp[1] - cy, 0.0])
    dr_len = np.linalg.norm(dr)
    r_world = dr / dr_len if dr_len > 1e-10 else np.array([1., 0., 0.])

    n_hat = np.cross(t_hat, r_world)
    n_len = np.linalg.norm(n_hat)
    if n_len < 1e-10:          # strut is purely radial — fall back to z-axis
        n_hat = np.cross(t_hat, np.array([0., 0., 1.]))
        n_len = np.linalg.norm(n_hat)
    n_hat /= max(n_len, 1e-10)

    r_hat  = np.cross(n_hat, t_hat)
    r_hat /= max(np.linalg.norm(r_hat), 1e-10)
    return r_hat, n_hat


def build_vertex_local_coords(mesh, network, center_xy):
    """
    For every mesh vertex find the nearest point on any strut centreline
    segment and express that vertex in the strut's local frame.

    Fully vectorised: O(V × E) but with numpy inner loops, typically < 1 s.

    Returns
    -------
    dict with keys
      edges    : list[(u,v)]      E edge tuples
      edge_idx : (V,) int32       which edge owns each vertex
      t_param  : (V,) float64     position in [0,1] along that edge
      r_comp   : (V,) float64     thickness offset  (r_hat direction)
      n_comp   : (V,) float64     width offset      (n_hat direction)
      t_comp   : (V,) float64     along-strut offset
      cx, cy   : float
    """
    verts    = mesh.vertices.astype(np.float64)        # (V, 3)
    V        = len(verts)
    edges    = list(network.graph.edges())             # E tuples
    E        = len(edges)
    npos     = network.node_positions.astype(np.float64)  # (N_nodes, 3)
    cx, cy   = float(center_xy[0]), float(center_xy[1])

    seg_a      = np.array([npos[u] for u, v in edges])   # (E, 3)
    seg_b      = np.array([npos[v] for u, v in edges])   # (E, 3)
    seg_d      = seg_b - seg_a                            # (E, 3)
    seg_len    = np.linalg.norm(seg_d, axis=1)            # (E,)
    seg_t_hat  = seg_d / np.maximum(seg_len, 1e-10)[:, None]  # (E, 3)

    # ── Vectorised nearest-point-on-segment for ALL (V, E) pairs ─────────────
    # ap[v, e] = verts[v] - seg_a[e],  shape (V, E, 3)
    ap    = verts[:, None, :] - seg_a[None, :, :]             # (V, E, 3)
    # Projection length along each segment (not yet clamped)
    proj  = np.einsum('ijk,jk->ij', ap, seg_t_hat)            # (V, E)
    # Clamp to [0, seg_len]
    t_raw = np.clip(proj, 0.0, seg_len[None, :])              # (V, E)
    # Nearest point on each segment
    near  = seg_a[None, :, :] + t_raw[:, :, None] * seg_t_hat[None, :, :]  # (V, E, 3)
    dist2 = np.einsum('ijk,ijk->ij', verts[:, None, :] - near,
                                     verts[:, None, :] - near)              # (V, E)

    # Best edge per vertex
    ei    = np.argmin(dist2, axis=1)                           # (V,)
    vi    = np.arange(V)
    t_abs = t_raw[vi, ei]                                      # (V,) absolute t along edge
    cp    = near[vi, ei]                                       # (V, 3) centreline points
    th    = seg_t_hat[ei]                                      # (V, 3) strut tangents

    # ── Per-vertex local frames (vectorised) ──────────────────────────────────
    dr     = cp[:, :2] - np.array([[cx, cy]])                 # (V, 2)
    dr_len = np.linalg.norm(dr, axis=1)                       # (V,)
    r_w2   = np.where(dr_len[:, None] > 1e-10,
                      dr / np.maximum(dr_len[:, None], 1e-10),
                      np.array([[1., 0.]]))                   # (V, 2)
    r_world = np.hstack([r_w2, np.zeros((V, 1))])             # (V, 3)

    n_hat  = np.cross(th, r_world)                            # (V, 3)
    n_len  = np.linalg.norm(n_hat, axis=1)
    radial = n_len < 1e-10
    if radial.any():
        z3                = np.zeros((V, 3)); z3[:, 2] = 1.
        n_hat[radial]     = np.cross(th[radial], z3[radial])
        n_len             = np.linalg.norm(n_hat, axis=1)
    n_hat /= np.maximum(n_len, 1e-10)[:, None]

    r_hat  = np.cross(n_hat, th)                              # (V, 3)
    r_hat /= np.maximum(np.linalg.norm(r_hat, axis=1), 1e-10)[:, None]

    # ── Local coordinates ─────────────────────────────────────────────────────
    delta  = verts - cp                                        # (V, 3)
    t_comp = np.einsum('ij,ij->i', delta, th)
    n_comp = np.einsum('ij,ij->i', delta, n_hat)
    r_comp = np.einsum('ij,ij->i', delta, r_hat)
    t_param = t_abs / np.maximum(seg_len[ei], 1e-10)          # normalise → [0,1]

    return {
        'edges':    edges,
        'edge_idx': ei.astype(np.int32),
        't_param':  t_param,
        'r_comp':   r_comp,
        'n_comp':   n_comp,
        't_comp':   t_comp,
        'cx': cx, 'cy': cy,
    }


def reconstruct_vertices_from_local_coords(lc, node_positions):
    """
    Rebuild world-space vertex positions from local-frame offsets and
    the deformed beam-node positions.

    lc             : dict returned by build_vertex_local_coords()
    node_positions : (N_nodes, 3) array of deformed skeleton positions
    """
    edges  = lc['edges']
    cx, cy = lc['cx'], lc['cy']
    npos   = np.asarray(node_positions, dtype=np.float64)

    seg_a  = np.array([npos[u] for u, v in edges])
    seg_b  = np.array([npos[v] for u, v in edges])
    seg_d  = seg_b - seg_a
    seg_len = np.linalg.norm(seg_d, axis=1)
    seg_t_hat = seg_d / np.maximum(seg_len, 1e-10)[:, None]

    ei = lc['edge_idx']
    tp = lc['t_param']
    rc = lc['r_comp']
    nc = lc['n_comp']
    tc = lc['t_comp']
    V  = len(ei)

    cp = seg_a[ei] + (tp * seg_len[ei])[:, None] * seg_t_hat[ei]  # (V, 3)
    th = seg_t_hat[ei]                                              # (V, 3)

    dr     = cp[:, :2] - np.array([[cx, cy]])
    dr_len = np.linalg.norm(dr, axis=1)
    r_w2   = np.where(dr_len[:, None] > 1e-10,
                      dr / np.maximum(dr_len[:, None], 1e-10),
                      np.array([[1., 0.]]))
    r_world = np.hstack([r_w2, np.zeros((V, 1))])

    n_hat  = np.cross(th, r_world)
    n_len  = np.linalg.norm(n_hat, axis=1)
    radial = n_len < 1e-10
    if radial.any():
        z3 = np.zeros((V, 3)); z3[:, 2] = 1.
        n_hat[radial] = np.cross(th[radial], z3[radial])
        n_len = np.linalg.norm(n_hat, axis=1)
    n_hat /= np.maximum(n_len, 1e-10)[:, None]

    r_hat  = np.cross(n_hat, th)
    r_hat /= np.maximum(np.linalg.norm(r_hat, axis=1), 1e-10)[:, None]

    return (cp + rc[:, None] * r_hat
               + nc[:, None] * n_hat
               + tc[:, None] * th).astype(np.float32)


def validate_cross_section_preservation(mesh, network, solver_frames, solver_meta,
                                         verbose=True):
    """
    STEP 1 VALIDATION

    Selects the most geometrically compact strut (edge whose assigned vertices
    cluster tightest around the centreline) and measures its cross-section WIDTH
    (n_hat direction) at t=0.25, 0.50, 0.75 along the strut in four states:
      Natural state / Full crimp / Mid deployment / Full deployment

    Only vertices within PROX_THRESH of the centreline are used so we measure a
    single cross-section, not a broad sweep of the mesh surface.

    All widths must agree to within 0.01 mm.  Prints results to console.
    Returns True if validation passes.
    """
    SEP = "=" * 64
    print(); print(SEP)
    print("STEP 1 VALIDATION — Cross-section frame preservation")
    print(SEP)

    center_xy = solver_meta[0]['center_xy']
    cx, cy    = float(center_xy[0]), float(center_xy[1])

    print("[validate] Building vertex local frames (natural state)...")
    lc = build_vertex_local_coords(mesh, network, center_xy)
    edges    = lc['edges']
    edge_idx = lc['edge_idx']
    t_param  = lc['t_param']
    r_comp   = lc['r_comp']
    n_comp   = lc['n_comp']
    V        = len(edge_idx)
    print(f"[validate] {V} vertices → {len(edges)} edges")

    # ── Radial distance from centreline in the cross-section plane ────────────
    cs_dist = np.sqrt(r_comp**2 + n_comp**2)   # (V,)

    # ── Select most-compact strut ─────────────────────────────────────────────
    # For every edge compute the 80th-percentile cs_dist of its assigned vertices.
    # The edge with the SMALLEST value has the most tightly-bound assignment
    # (i.e. its vertices are genuinely on that strut's cross-section).
    npos_nat = network.node_positions.astype(np.float64)
    MIN_VERTS_FOR_CANDIDATE = 8
    T_SPAN_MIN = 0.30        # close vertices must span ≥30% of the edge length

    edge_p80    = np.full(len(edges), np.inf)
    edge_t_span = np.zeros(len(edges))

    for ei_cand in range(len(edges)):
        m_cand = edge_idx == ei_cand
        if m_cand.sum() < MIN_VERTS_FOR_CANDIDATE:
            continue
        p80 = float(np.percentile(cs_dist[m_cand], 80))
        edge_p80[ei_cand] = p80
        # t-span of the close vertices for this edge (provisional threshold)
        close_cand = m_cand & (cs_dist < p80 * 1.5)
        if close_cand.sum() >= MIN_VERTS_FOR_CANDIDATE:
            t_c = t_param[close_cand]
            edge_t_span[ei_cand] = float(t_c.max() - t_c.min())

    # Select most-compact edge that also has good axial vertex spread.
    # Fall back to looser thresholds if no edge qualifies.
    best_ei = None
    used_t_thresh = 0.0
    for t_thresh in [T_SPAN_MIN, 0.15, 0.05, 0.0]:
        valid = (edge_p80 < np.inf) & (edge_t_span >= t_thresh)
        if valid.any():
            best_ei = int(np.argmin(np.where(valid, edge_p80, np.inf)))
            used_t_thresh = t_thresh
            break

    PROX_THRESH = edge_p80[best_ei] * 1.5

    u, v        = edges[best_ei]
    seg_nat     = npos_nat[v] - npos_nat[u]
    seg_len_nat = float(np.linalg.norm(seg_nat))
    all_on_edge = edge_idx == best_ei
    close_mask  = all_on_edge & (cs_dist < PROX_THRESH)   # ← proximity filter
    n_close     = int(close_mask.sum())

    print(f"[validate] Chosen strut: edge {best_ei} (nodes {u}→{v}), "
          f"natural length = {seg_len_nat:.3f} mm  "
          f"t_span={edge_t_span[best_ei]:.3f} (thresh={used_t_thresh:.2f})")
    print(f"[validate] Proximity threshold = {PROX_THRESH:.4f} mm  "
          f"({n_close} of {int(all_on_edge.sum())} assigned vertices pass)")

    if n_close < 4:
        print("[validate] Too few vertices in proximity window — skipping validation.")
        print(SEP); print()
        return False

    t_s = t_param[close_mask]   # (K,) t_params of close vertices

    # Report natural cross-section dimensions
    r_on = r_comp[close_mask]; n_on = n_comp[close_mask]
    print(f"[validate] Natural cross-section  "
          f"width (n_hat) = {n_on.max()-n_on.min():.4f} mm  "
          f"thick (r_hat) = {r_on.max()-r_on.min():.4f} mm")
    print()

    t_positions = [0.25, 0.50, 0.75]
    window      = 0.15   # ± half-window in t_param

    # ── Deformation states ────────────────────────────────────────────────────
    n_crimp = sum(1 for m in solver_meta if m['type'] == 'crimp')
    n_total = len(solver_frames)
    states  = [
        ("Natural",     None),
        ("Full crimp",  n_crimp - 1),
        ("Mid deploy",  n_crimp + (n_total - n_crimp) // 2),
        ("Full deploy", n_total - 1),
    ]

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"  {'State':<14}  {'t=0.25 (mm)':>12}  {'t=0.50 (mm)':>12}  "
          f"{'t=0.75 (mm)':>12}  {'pass?':>6}")
    print(f"  {'-'*14}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*6}")

    all_widths = []
    nat_widths = None

    def _measure_widths(verts_close, node_pos_state):
        """Measure n_hat width at each t_position for the given deformed vertices."""
        npos  = node_pos_state.astype(np.float64)
        a     = npos[u]; b = npos[v]
        seg   = b - a
        sl    = max(float(np.linalg.norm(seg)), 1e-10)
        t_hat = seg / sl
        ws = []
        for t_pos in t_positions:
            in_win = np.abs(t_s - t_pos) < window
            if in_win.sum() < 2:
                ws.append(float('nan'))
                continue
            cp        = a + t_pos * seg
            _, n_hat  = _local_frame_at(cp, t_hat, cx, cy)
            delta     = verts_close[in_win].astype(np.float64) - cp[None, :]
            n_vals    = delta @ n_hat
            ws.append(float(n_vals.max() - n_vals.min()))
        return ws

    for label, fi in states:
        if fi is None:
            verts_c     = mesh.vertices[close_mask]
            node_pos_st = npos_nat
        else:
            all_v       = reconstruct_vertices_from_local_coords(lc, solver_frames[fi])
            verts_c     = all_v[close_mask]
            node_pos_st = solver_frames[fi]

        ws = _measure_widths(verts_c, node_pos_st)
        all_widths.append(ws)
        if fi is None:
            nat_widths = ws

        if fi is None:
            flag = "  ref"
        else:
            diffs = [abs(ws[i] - nat_widths[i])
                     for i in range(3)
                     if not np.isnan(ws[i]) and not np.isnan(nat_widths[i])]
            flag  = " ✓ ok" if diffs and all(d < 0.01 for d in diffs) else " ✗ FAIL"

        row = "  ".join(f"{w:>12.4f}" if not np.isnan(w) else f"{'N/A':>12}" for w in ws)
        print(f"  {label:<14}  {row}  {flag}")

    # ── OLD method comparison (full crimp only) ───────────────────────────────
    print()
    print("  Old cylindrical method for comparison (full crimp):")
    fi_crimp  = n_crimp - 1
    scale_c   = float(solver_meta[fi_crimp]['scale'])
    v_orig    = mesh.vertices.astype(np.float64)
    cyl_v     = v_orig.copy()
    cyl_v[:, 0] = cx + (v_orig[:, 0] - cx) * scale_c
    cyl_v[:, 1] = cy + (v_orig[:, 1] - cy) * scale_c
    ws_old = _measure_widths(cyl_v[close_mask], solver_frames[fi_crimp])
    row_old = "  ".join(f"{w:>12.4f}" if not np.isnan(w) else f"{'N/A':>12}" for w in ws_old)
    print(f"  {'Cyl (old)':<14}  {row_old}  (cylindrical scaling shrinks cross-section)")

    # ── Pass / fail ───────────────────────────────────────────────────────────
    passed = True
    print()
    for ti, t_pos in enumerate(t_positions):
        ref = nat_widths[ti] if nat_widths else float('nan')
        if np.isnan(ref):
            continue
        for si, (label, fi) in enumerate(states[1:], 1):
            w    = all_widths[si][ti]
            if np.isnan(w): continue
            diff = abs(w - ref)
            if diff >= 0.01:
                passed = False
                print(f"  FAIL  t={t_pos:.2f}  {label}: "
                      f"width={w:.4f}  ref={ref:.4f}  diff={diff:.4f} mm")

    print()
    print(f"Result: {'PASSED ✓' if passed else 'FAILED ✗'}")
    print(SEP); print()
    return passed


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
