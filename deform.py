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
    For every mesh vertex find the nearest point on the DENSIFIED strut
    centreline skeleton and store the 3D world-space offset from that point.

    Densifies each skeleton edge to < MAX_DENSE_SEG_LEN mm before binding so
    neighbouring struts (mm apart) cannot steal vertices from each other.
    Uses a KD-tree on segment midpoints for fast candidate lookup.

    Returns
    -------
    dict with keys
      edges       : list[(u,v)]    E original edge tuples
      npos_nat    : (N_nodes, 3)   natural skeleton node positions
      edge_idx    : (V,) int32     which original edge owns each vertex
      t_param     : (V,) float64   position in [0,1] along original edge
      offset      : (V, 3) float64 world-space offset: vertex − centreline_nat
      r_comp      : (V,) float64   thickness component (validation compat)
      n_comp      : (V,) float64   width component     (validation compat)
      t_comp      : (V,) float64   along-strut component
      max_bind_dist : float        max |offset| across all vertices
      cx, cy      : float
    """
    from scipy.spatial import cKDTree

    MAX_DENSE_SEG_LEN = 0.3   # mm — must be << strut spacing (several mm)

    verts  = mesh.vertices.astype(np.float64)
    V      = len(verts)
    edges  = list(network.graph.edges())
    E      = len(edges)
    npos   = network.node_positions.astype(np.float64)
    cx, cy = float(center_xy[0]), float(center_xy[1])

    # ── Step 1: Densify each skeleton edge ────────────────────────────────────
    d_a_list, d_b_list, d_orig, d_ta, d_tb = [], [], [], [], []
    for ei, (u, v) in enumerate(edges):
        seg   = npos[v] - npos[u]
        seg_l = float(np.linalg.norm(seg))
        n_sub = max(1, int(np.ceil(seg_l / MAX_DENSE_SEG_LEN)))
        ts    = np.linspace(0.0, 1.0, n_sub + 1)
        for k in range(n_sub):
            ta, tb = float(ts[k]), float(ts[k + 1])
            d_a_list.append(npos[u] + ta * seg)
            d_b_list.append(npos[u] + tb * seg)
            d_orig.append(ei)
            d_ta.append(ta)
            d_tb.append(tb)

    dseg_a = np.array(d_a_list, dtype=np.float64)   # (D, 3)
    dseg_b = np.array(d_b_list, dtype=np.float64)   # (D, 3)
    d_orig = np.array(d_orig,   dtype=np.int32)      # (D,)
    d_ta   = np.array(d_ta,     dtype=np.float64)    # (D,)
    d_tb   = np.array(d_tb,     dtype=np.float64)    # (D,)
    D      = len(dseg_a)

    print(f"[bind] Dense skeleton: {D} segs from {E} orig edges "
          f"(max_len={MAX_DENSE_SEG_LEN:.2f} mm)")

    # ── Step 2: KD-tree on segment midpoints, query K candidates per vertex ───
    seg_mid     = 0.5 * (dseg_a + dseg_b)
    tree        = cKDTree(seg_mid)
    K_CAND      = min(16, D)
    _, idx_cand = tree.query(verts, k=K_CAND)        # (V, K_CAND)

    # ── Step 3: Exact dist to each candidate → pick nearest ──────────────────
    best_dist2      = np.full(V, np.inf)
    best_ds         = np.zeros(V, dtype=np.int32)
    best_t_in_dense = np.zeros(V, dtype=np.float64)

    for ki in range(K_CAND):
        ds  = idx_cand[:, ki]
        a   = dseg_a[ds]
        b   = dseg_b[ds]
        ab  = b - a
        ab2 = np.einsum('ij,ij->i', ab, ab)
        t_r = np.einsum('ij,ij->i', verts - a, ab) / np.maximum(ab2, 1e-20)
        t_c = np.clip(t_r, 0.0, 1.0)
        cl  = a + t_c[:, None] * ab
        d2  = np.einsum('ij,ij->i', verts - cl, verts - cl)
        upd = d2 < best_dist2
        best_dist2[upd]      = d2[upd]
        best_ds[upd]         = ds[upd]
        best_t_in_dense[upd] = t_c[upd]

    # ── Step 4: Map dense assignment back to original edge + t_param ──────────
    ei_v      = d_orig[best_ds]
    t_in_orig = d_ta[best_ds] + best_t_in_dense * (d_tb[best_ds] - d_ta[best_ds])

    # ── Step 5: Natural-state centreline points and offsets ──────────────────
    seg_a     = np.array([npos[u] for u, v in edges])
    seg_b     = np.array([npos[v] for u, v in edges])
    seg_d     = seg_b - seg_a
    seg_len   = np.linalg.norm(seg_d, axis=1)
    seg_t_hat = seg_d / np.maximum(seg_len, 1e-10)[:, None]

    cp_nat  = seg_a[ei_v] + t_in_orig[:, None] * seg_d[ei_v]
    offsets = verts - cp_nat                                     # (V, 3)

    # ── Step 6: Decompose offsets for validation compatibility ────────────────
    th     = seg_t_hat[ei_v]
    t_comp = np.einsum('ij,ij->i', offsets, th)

    dr     = cp_nat[:, :2] - np.array([[cx, cy]])
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

    r_comp = np.einsum('ij,ij->i', offsets, r_hat)
    n_comp = np.einsum('ij,ij->i', offsets, n_hat)

    # ── Binding quality report ────────────────────────────────────────────────
    dist_cl = np.sqrt(best_dist2)
    max_bd  = float(dist_cl.max())
    print(f"[bind] Binding: max={max_bd:.3f} mm  mean={dist_cl.mean():.3f} mm  "
          f">1 mm: {(dist_cl > 1.0).sum()}  >0.5 mm: {(dist_cl > 0.5).sum()}")
    if max_bd > 1.0:
        print(f"[bind] WARNING: {(dist_cl > 1.0).sum()} vertices bound > 1 mm from "
              "centreline — reduce MAX_DENSE_SEG_LEN if this number is large")

    return {
        'edges':         edges,
        'npos_nat':      npos.copy(),
        'edge_idx':      ei_v.astype(np.int32),
        't_param':       t_in_orig,
        'offset':        offsets,
        'r_comp':        r_comp,
        'n_comp':        n_comp,
        't_comp':        t_comp,
        'max_bind_dist': max_bd,
        'cx': cx, 'cy': cy,
    }


def reconstruct_vertices_from_local_coords(lc, node_positions):
    """
    Rebuild world-space vertex positions by translating each vertex by its
    skeleton centreline's displacement:

        new_vertex = vertex_nat + (cp_def - cp_nat)
                   = cp_def + offset_nat

    No rotation needed: r_hat direction is preserved by radial deformation,
    so the natural-state offset is already in the correct frame.  Rodrigues
    rotation was tried but causes explosions when skeleton tangents change
    drastically across the deployment release front.

    lc             : dict returned by build_vertex_local_coords()
    node_positions : (N_nodes, 3) deformed skeleton node positions
    """
    edges    = lc['edges']
    npos_def = np.asarray(node_positions, dtype=np.float64)

    # Deformed skeleton geometry
    seg_a_d = np.array([npos_def[u] for u, v in edges])   # (E, 3)
    seg_b_d = np.array([npos_def[v] for u, v in edges])   # (E, 3)
    seg_d_d = seg_b_d - seg_a_d                            # (E, 3)

    ei      = lc['edge_idx']   # (V,) int32
    tp      = lc['t_param']    # (V,) float64  ∈ [0, 1]
    offsets = lc['offset']     # (V, 3) float64  vertex_nat - cp_nat

    # Deformed centreline point for each vertex's assigned edge segment
    cp_def = seg_a_d[ei] + tp[:, None] * seg_d_d[ei]   # (V, 3)

    # Translate: add unchanged natural-state offset (preserves cross-section)
    return (cp_def + offsets).astype(np.float32)


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
    print(f"[validate] {V} vertices -> {len(edges)} edges")

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

    print(f"[validate] Chosen strut: edge {best_ei} (nodes {u}->{v}), "
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

    # Adaptive measurement positions — derived from where close vertices actually sit,
    # so windows always contain vertices regardless of how the edge is parameterised.
    t_positions = sorted([
        float(np.percentile(t_s, 20)),
        float(np.percentile(t_s, 50)),
        float(np.percentile(t_s, 80)),
    ])
    t_spread = t_positions[2] - t_positions[0]
    window   = max(t_spread * 0.35, 0.04)
    if t_spread < 0.10:
        print(f"[validate] Warning: close vertices span only {t_spread:.3f} of the edge "
              f"(t=[{t_positions[0]:.3f},{t_positions[2]:.3f}]) — "
              "measuring at a single cross-section section only")

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

    # Pre-compute natural-state n_hat for each t_position so the same frame
    # is used across all deformation states.  Using the deformed n_hat would
    # make the width direction state-dependent and produce false FAIL reports
    # (e.g. a diagonal strut that tilts during radial crimp changes t_hat,
    # rotating n_hat, which alters the measured projection even when the
    # physical cross-section is unchanged).
    nat_npos = npos_nat.astype(np.float64)
    nat_a = nat_npos[u]; nat_b = nat_npos[v]
    nat_seg = nat_b - nat_a
    nat_sl  = max(float(np.linalg.norm(nat_seg)), 1e-10)
    nat_t_hat = nat_seg / nat_sl
    nat_n_hats = []
    nat_cps    = []
    for t_pos in t_positions:
        cp_n = nat_a + t_pos * nat_seg
        _, n_h = _local_frame_at(cp_n, nat_t_hat, cx, cy)
        nat_n_hats.append(n_h)
        nat_cps.append(cp_n)

    def _measure_widths(verts_close, node_pos_state):
        """Measure n_hat width at each t_position using the fixed natural-state frame."""
        ws = []
        for i, t_pos in enumerate(t_positions):
            in_win = np.abs(t_s - t_pos) < window
            if in_win.sum() < 2:
                ws.append(float('nan'))
                continue
            n_hat  = nat_n_hats[i]
            cp     = nat_cps[i]
            delta  = verts_close[in_win].astype(np.float64) - cp[None, :]
            n_vals = delta @ n_hat
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
            flag  = "   ok " if diffs and all(d < 0.01 for d in diffs) else " FAIL "

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
    print(f"Result: {'PASSED' if passed else 'FAILED'}")
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

    # ── Precompute cylindrical coords ────────────────────────────────────────
    dx = orig[:, 0] - cx;  dy = orig[:, 1] - cy
    r_orig   = np.sqrt(dx**2 + dy**2)
    theta    = np.arctan2(dy, dx)
    z_orig   = orig[:, 2]
    r_center = np.median(r_orig)
    r_offset = r_orig - r_center

    # ── Skeleton binding — to identify long axial strut vertices ─────────────
    # We bind vertices to skeleton edges so we know which ones belong to long
    # nearly-axial struts.  These struts get an additive bow correction on top
    # of the base per-vertex formula (endpoints unchanged → no boundary artifact).
    lc_export = build_vertex_local_coords(mesh, network, cxy)
    _npos     = network.node_positions.astype(np.float64)
    _edges    = list(network.graph.edges())
    _cs_dist  = np.sqrt(lc_export['r_comp']**2 + lc_export['n_comp']**2)

    AXIAL_MIN_DZ      = 30.0   # mm — z-span for "long" strut
    AXIAL_MAX_DXY_RAT = 0.12   # max XY drift / dz for "nearly axial"
    AXIAL_BIND_MAX    = 1.0    # mm — max binding dist (≈ strut radius)
    BOW_FRAC          = 0.15   # bow amplitude as fraction of endpoint r-delta

    axial_strut_list = []
    for ei, (u, v) in enumerate(_edges):
        dz  = abs(_npos[v, 2] - _npos[u, 2])
        dxy = float(np.linalg.norm(_npos[v, :2] - _npos[u, :2]))
        if dz < AXIAL_MIN_DZ or dxy > dz * AXIAL_MAX_DXY_RAT:
            continue
        v_mask = (lc_export['edge_idx'] == ei) & (_cs_dist < AXIAL_BIND_MAX)
        if v_mask.sum() < 8:
            continue
        axial_strut_list.append({'mask': v_mask, 'u': u, 'v': v,
                                  't': lc_export['t_param'][v_mask]})

    # ── Crown geometry for dwell ──────────────────────────────────────────────
    node_z      = network.node_positions[:, 2]
    z_span_mesh = float(z_orig.max() - z_orig.min())
    cluster_tol = 0.05 * z_span_mesh
    z_sorted = np.sort(node_z)
    cluster_centers = [float(z_sorted[0])]
    for zv in z_sorted[1:]:
        if zv - cluster_centers[-1] > cluster_tol:
            cluster_centers.append(float(zv))
        else:
            cluster_centers[-1] = 0.5 * (cluster_centers[-1] + float(zv))
    n_cells = max(1, (len(cluster_centers) - 1) // 2)
    crown_arm_length = z_span_mesh / (2.0 * n_cells)
    cell_height = 2.0 * crown_arm_length

    z_in_cell        = (z_orig - float(z_orig.min())) % cell_height
    crown_proximity  = 2.0 * np.abs(z_in_cell / cell_height - 0.5)
    dwell_per_vertex = crown_arm_length * crown_dwell * crown_proximity
    _deploy_eff_zmin = float((z_orig - dwell_per_vertex).min())

    if verbose:
        print(f"[deform] Input: {n_verts} verts, {n_faces} faces")
        print(f"[deform] Center: ({cx:.2f}, {cy:.2f})  "
              f"R range: [{r_orig.min():.2f}, {r_orig.max():.2f}] mm")
        print(f"[deform] Crown: {n_cells} cells, arm={crown_arm_length:.2f} mm")

    # ── Per-frame deformation ─────────────────────────────────────────────────
    paths: List[str] = []

    for idx, fm in tqdm(enumerate(meta), total=len(meta),
                        desc="Exporting STL frames", disable=not verbose):

        crimp_r  = fm['crimp_r']
        deploy_r = fm['deploy_r']

        # ──────────────────────────────────────────────────────────────────────
        # CRIMP — pure cylindrical radial scale, Z frozen
        # ──────────────────────────────────────────────────────────────────────
        if fm['type'] == 'crimp':
            r_new = r_center * fm['scale'] + r_offset

        # ──────────────────────────────────────────────────────────────────────
        # DEPLOY — pure per-vertex cylindrical release
        # ──────────────────────────────────────────────────────────────────────
        else:
            z_min  = fm['z_min'];   z_span = fm['z_span']
            z_max  = z_min + z_span;  z_front = fm['z_front_norm']
            trans_len     = transition_frac * z_span
            deploy_travel = z_max - (_deploy_eff_zmin - trans_len)
            tube_tip_z    = z_max - z_front * deploy_travel

            z_eff_v    = z_orig - dwell_per_vertex
            released_v = _smoothstep((z_eff_v - tube_tip_z) / trans_len)
            snap_v     = _snap_curve(released_v, snap_speed)
            r_new      = crimp_r + (deploy_r - crimp_r) * snap_v + r_offset

        # ── Reconstruct vertices (pure cylindrical — both crimp and deploy) ───
        new_verts = np.empty_like(orig)
        new_verts[:, 0] = cx + r_new * np.cos(theta)
        new_verts[:, 1] = cy + r_new * np.sin(theta)
        new_verts[:, 2] = z_orig

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
