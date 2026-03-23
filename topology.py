"""
topology.py — STL parsing and strut/beam network extraction.

Strategy:
1. Load the STL mesh via trimesh.
2. Extract the mesh's unique edges (each edge shared by exactly 2 faces = interior;
   boundary edges or all edges depending on mesh type).
3. Cluster short nearly-collinear edge chains into beam segments.
4. Detect junction nodes (degree > 2) and endpoint nodes (degree 1).
5. Build a NetworkX graph: nodes = 3-D positions, edges = beam elements.

Fallback: if skeletonization fails (e.g. mesh is closed/watertight solid), fall
back to voxel-skeleton via trimesh.voxel or a centerline approximation.
"""

import numpy as np
import trimesh
import networkx as nx
from scipy.spatial import KDTree
from utils import bounding_box, centroid

# ── tuneable parameters ────────────────────────────────────────────────────────
MERGE_RADIUS_FRACTION = 0.02   # merge nodes closer than this fraction of bbox diagonal
MIN_BEAM_LENGTH_FRACTION = 0.005  # discard edges shorter than this fraction of diagonal
MAX_COLLINEAR_ANGLE_DEG = 15.0    # chain edges if angle between them < this threshold
# ──────────────────────────────────────────────────────────────────────────────


# ── mesh preprocessing ─────────────────────────────────────────────────────────

def preprocess_mesh(mesh: trimesh.Trimesh,
                    deployed_diameter_mm: float = None,
                    verbose: bool = True) -> trimesh.Trimesh:
    """
    Standardise the mesh before topology extraction:
      1. Detect and convert units to mm.
      2. Reorient so the stent long axis aligns with +Z.
      3. Centre the stent at the origin (XY) and bottom at Z=0.

    Unit detection heuristic:
      - If longest extent > 200  → assume micrometres (÷1000)
      - If longest extent > 50   → assume already in mm (÷1)
      - If longest extent > 5    → assume cm (×10)
      - Otherwise                → assume inches (×25.4)
      If deployed_diameter_mm is supplied, that cross-check is used instead.
    """
    extents = mesh.extents
    longest = extents.max()

    # ── Unit detection ────────────────────────────────────────────────────────
    if deployed_diameter_mm is not None:
        # The two shorter axes approximate the stent diameter.
        # Use the median of the two shorter extents as the cross-section size.
        sorted_ext = np.sort(extents)
        cross_section = np.mean(sorted_ext[:2])          # average of two shorter axes
        scale = deployed_diameter_mm / cross_section
    else:
        if longest > 200:
            scale = 0.001        # micrometres → mm
        elif longest > 50:
            scale = 1.0          # already mm
        elif longest > 5:
            scale = 10.0         # cm → mm
        else:
            scale = 25.4         # inches → mm

    if abs(scale - 1.0) > 0.01:
        if verbose:
            print(f"[topology] Unit conversion: scale x{scale:.4f} "
                  f"(longest extent {longest:.4f} -> {longest*scale:.1f} mm)")
        mesh = mesh.copy()
        mesh.apply_scale(scale)

    # ── Reorient long axis to Z ───────────────────────────────────────────────
    extents2 = mesh.extents
    long_ax  = int(np.argmax(extents2))   # 0=X, 1=Y, 2=Z
    ax_names = ['X', 'Y', 'Z']

    if long_ax != 2:
        if verbose:
            print(f"[topology] Long axis is {ax_names[long_ax]} "
                  f"({extents2[long_ax]:.1f} mm) — rotating to Z.")
        # Build rotation that maps long_ax → Z
        if long_ax == 0:      # X → Z: rotate -90° around Y
            R = trimesh.transformations.rotation_matrix(-np.pi/2, [0, 1, 0])
        else:                  # Y → Z: rotate +90° around X
            R = trimesh.transformations.rotation_matrix( np.pi/2, [1, 0, 0])
        mesh = mesh.copy()
        mesh.apply_transform(R)

    # ── Centre: XY at origin, Z bottom at 0 ──────────────────────────────────
    c = mesh.centroid
    t = np.eye(4)
    t[0, 3] = -c[0]
    t[1, 3] = -c[1]
    t[2, 3] = -mesh.bounds[0, 2]    # shift bottom to Z=0
    mesh = mesh.copy()
    mesh.apply_transform(t)

    if verbose:
        print(f"[topology] After preprocessing: extents={mesh.extents.round(2)} mm, "
              f"bounds Z=[{mesh.bounds[0,2]:.2f}, {mesh.bounds[1,2]:.2f}]")

    return mesh


class BeamNetwork:
    """
    Container for the beam-node graph derived from an STL stent mesh.

    Attributes
    ----------
    graph : nx.Graph
        Nodes carry attribute 'pos' (np.ndarray shape (3,)).
        Edges carry attribute 'rest_length' (float) and 'radius' (float).
    node_positions : np.ndarray  shape (N, 3)
    node_ids       : list of node keys (integers)
    """

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self._reindex()

    def _reindex(self):
        mapping = {old: new for new, old in enumerate(sorted(self.graph.nodes()))}
        self.graph = nx.relabel_nodes(self.graph, mapping)
        self.node_ids = list(self.graph.nodes())
        self.node_positions = np.array([self.graph.nodes[n]['pos'] for n in self.node_ids])

    @property
    def num_nodes(self):
        return len(self.node_ids)

    @property
    def num_beams(self):
        return self.graph.number_of_edges()

    def beam_list(self):
        """Return list of (i, j) node-index tuples."""
        return [(u, v) for u, v in self.graph.edges()]

    def beam_radii(self):
        """Return array of strut radii, shape (num_beams,)."""
        return np.array([self.graph[u][v].get('radius', 0.1) for u, v in self.graph.edges()])

    def __repr__(self):
        return f"<BeamNetwork nodes={self.num_nodes} beams={self.num_beams}>"


# ── main entry point ───────────────────────────────────────────────────────────

def load_and_extract(stl_path: str, verbose: bool = True,
                     deployed_diameter_mm: float = None) -> tuple[trimesh.Trimesh, BeamNetwork]:
    """
    Load an STL file and extract a beam-node network.

    Returns
    -------
    mesh : trimesh.Trimesh
    network : BeamNetwork
    """
    if verbose:
        print(f"[topology] Loading STL: {stl_path}")
    mesh = trimesh.load(stl_path, force='mesh')

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load a single mesh from {stl_path!r}. "
                         "Make sure the file contains exactly one solid body.")

    if verbose:
        print(f"[topology] Mesh: {len(mesh.vertices)} vertices, "
              f"{len(mesh.faces)} faces, watertight={mesh.is_watertight}")

    # ── Preprocessing: units, orientation, centering ──────────────────────────
    mesh = preprocess_mesh(mesh, deployed_diameter_mm=deployed_diameter_mm,
                           verbose=verbose)

    bb_min, bb_max = bounding_box(mesh.vertices)
    diag = np.linalg.norm(bb_max - bb_min)
    merge_r = MERGE_RADIUS_FRACTION * diag
    min_len  = MIN_BEAM_LENGTH_FRACTION * diag

    # ── try skeleton-based extraction first ───────────────────────────────────
    network = None
    try:
        network = _extract_via_skeleton(mesh, merge_r, min_len, verbose)
    except Exception as e:
        if verbose:
            print(f"[topology] Skeleton extraction failed ({e}); "
                  "falling back to edge-based extraction.")

    if network is None or network.num_beams < 4:
        try:
            network = _extract_via_edges(mesh, merge_r, min_len, verbose)
        except Exception as e:
            if verbose:
                print(f"[topology] Edge extraction failed ({e}); "
                      "using analytical radial-lattice fallback.")
            network = _analytical_fallback(mesh, verbose)

    if verbose:
        print(f"[topology] {network}")

    return mesh, network


# ── skeleton-based extraction ──────────────────────────────────────────────────

def _extract_via_skeleton(mesh, merge_r, min_len, verbose):
    """
    Use trimesh's path extraction on the mesh boundary / edges to get centerlines.
    Works well for open-surface (shell) meshes like exported stent STLs.
    """
    # For shell meshes: extract unique boundary/free edges
    edges = _get_strut_edges(mesh)
    if len(edges) < 10:
        raise ValueError("Too few candidate strut edges found.")

    verts = mesh.vertices
    G = _build_raw_graph(verts, edges, merge_r, min_len)

    if verbose:
        print(f"[topology] Raw graph: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges")

    G = _simplify_graph(G, MAX_COLLINEAR_ANGLE_DEG)

    # Prune spurious long beams (longer than stent diameter × 0.7)
    bb_min2, bb_max2 = bounding_box(mesh.vertices)
    stent_extents = bb_max2 - bb_min2
    # Use the mean of the two cross-section extents as diameter
    cross_axes = np.sort(stent_extents)[:2]
    stent_diam = float(cross_axes.mean())
    max_beam_len = stent_diam * 0.7
    G = _prune_graph(G, max_beam_len, verbose=verbose)

    _assign_strut_radii(G, mesh)

    return BeamNetwork(G)


def _get_strut_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Return (M, 2) array of vertex-index pairs representing strut centerline edges.

    For open meshes: use edges that appear in an odd number of faces (boundary).
    For closed meshes: use ALL unique edges (works when the mesh is a thin shell).
    """
    # edges_unique gives each undirected edge once
    edges = mesh.edges_unique  # shape (E, 2)

    # Count how many faces reference each unique edge
    # edges_unique_inverse maps each face-edge back to unique edge index
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(edges))

    # Boundary edges appear once; interior edges appear twice
    boundary_mask = counts == 1
    if boundary_mask.sum() > 10:
        return edges[boundary_mask]

    # Fallback: all unique edges
    return edges


def _build_raw_graph(verts, edges, merge_r, min_len):
    """
    Build a networkx graph from vertex/edge arrays, merging nearby nodes.
    """
    # Merge nearby vertices
    tree = KDTree(verts)
    merge_map = np.arange(len(verts))
    visited = np.zeros(len(verts), dtype=bool)

    for i in range(len(verts)):
        if visited[i]:
            continue
        neighbours = tree.query_ball_point(verts[i], merge_r)
        for nb in neighbours:
            merge_map[nb] = i
            visited[nb] = True

    G = nx.Graph()
    for i, j in edges:
        ni, nj = merge_map[i], merge_map[j]
        if ni == nj:
            continue
        length = np.linalg.norm(verts[ni] - verts[nj])
        if length < min_len:
            continue
        if not G.has_node(ni):
            G.add_node(ni, pos=verts[ni].copy())
        if not G.has_node(nj):
            G.add_node(nj, pos=verts[nj].copy())
        if not G.has_edge(ni, nj):
            G.add_edge(ni, nj, rest_length=length)

    return G


def _simplify_graph(G: nx.Graph, max_angle_deg: float) -> nx.Graph:
    """
    Collapse degree-2 nodes that are collinear with their neighbours into
    a single straight beam edge (reduces DOF without losing topology).
    """
    cos_thresh = np.cos(np.radians(max_angle_deg))
    changed = True
    while changed:
        changed = False
        for node in list(G.nodes()):
            if G.degree(node) != 2:
                continue
            nbrs = list(G.neighbors(node))
            a, b = nbrs[0], nbrs[1]
            pa = G.nodes[a]['pos']
            pb = G.nodes[b]['pos']
            pc = G.nodes[node]['pos']
            va = pa - pc
            vb = pb - pc
            na = np.linalg.norm(va)
            nb_ = np.linalg.norm(vb)
            if na < 1e-12 or nb_ < 1e-12:
                continue
            cos_a = np.dot(va / na, vb / nb_)
            if cos_a < cos_thresh:   # nearly anti-parallel → collinear
                new_len = np.linalg.norm(pa - pb)
                G.add_edge(a, b, rest_length=new_len)
                G.remove_node(node)
                changed = True
                break

    return G


def _prune_graph(G: nx.Graph, max_beam_length: float,
                 verbose: bool = False) -> nx.Graph:
    """
    Remove beams longer than max_beam_length (spurious cross-stent edges),
    then iteratively prune degree-1 (dangling) nodes until stable.

    Long beams usually result from the edge-extraction picking up continuous
    boundary edges that run axially along a solid-mesh strut, creating
    node-to-node connections that span the full stent height.
    """
    # Remove over-long beams
    long_edges = [(u, v) for u, v, d in G.edges(data=True)
                  if d['rest_length'] > max_beam_length]
    if verbose and long_edges:
        print(f"[topology] Pruning {len(long_edges)} beams longer than "
              f"{max_beam_length:.1f} mm")
    G.remove_edges_from(long_edges)

    # Prune dangling nodes (degree 0 or 1) iteratively
    changed = True
    while changed:
        changed = False
        danglers = [n for n in list(G.nodes()) if G.degree(n) <= 1]
        if danglers:
            G.remove_nodes_from(danglers)
            changed = True

    # Remove isolated components smaller than 3 nodes
    large = max(nx.connected_components(G), key=len) if G.number_of_nodes() > 0 else set()
    small_nodes = [n for n in G.nodes() if n not in large]
    # Keep all components with >= 3 nodes
    for comp in list(nx.connected_components(G)):
        if len(comp) < 3:
            G.remove_nodes_from(comp)

    return G


def _assign_strut_radii(G: nx.Graph, mesh: trimesh.Trimesh):
    """
    Estimate strut cross-section radius for each beam edge.
    Uses the mesh's local thickness (half of nearest-surface distance).
    """
    bb_min, bb_max = bounding_box(mesh.vertices)
    default_r = np.linalg.norm(bb_max - bb_min) * 0.005  # 0.5% of diagonal

    try:
        # Sample midpoints of each edge and query mesh for proximity
        tree = KDTree(mesh.vertices)
        for u, v, data in G.edges(data=True):
            mid = 0.5 * (G.nodes[u]['pos'] + G.nodes[v]['pos'])
            dist, _ = tree.query(mid)
            data['radius'] = max(dist * 0.5, default_r * 0.5)
    except Exception:
        for _, _, data in G.edges(data=True):
            data['radius'] = default_r


# ── edge-based extraction (fallback 1) ────────────────────────────────────────

def _extract_via_edges(mesh, merge_r, min_len, verbose):
    """
    Simpler fallback: treat all unique edges of the mesh as candidate beams,
    then cluster to find the medial axis empirically.
    """
    if verbose:
        print("[topology] Using all-edge extraction strategy.")

    verts = mesh.vertices
    edges = mesh.edges_unique
    G = _build_raw_graph(verts, edges, merge_r * 2, min_len * 2)
    G = _simplify_graph(G, MAX_COLLINEAR_ANGLE_DEG * 2)
    bb_min2, bb_max2 = bounding_box(verts)
    stent_extents = bb_max2 - bb_min2
    cross_axes = np.sort(stent_extents)[:2]
    stent_diam = float(cross_axes.mean())
    G = _prune_graph(G, stent_diam * 0.7, verbose=verbose)
    _assign_strut_radii(G, mesh)
    return BeamNetwork(G)


# ── analytical radial-lattice fallback ────────────────────────────────────────

def _analytical_fallback(mesh: trimesh.Trimesh, verbose: bool) -> BeamNetwork:
    """
    When topology extraction is impossible (e.g. overly coarse or solid mesh),
    generate a parametric stent-like beam network that matches the mesh bounding
    cylinder: N_z rings × N_circ nodes, connected with zigzag struts.

    This gives the solver a valid beam network even when the mesh skeleton is
    unrecoverable.
    """
    if verbose:
        print("[topology] Building analytical fallback beam network.")

    bb_min, bb_max = bounding_box(mesh.vertices)
    center_xy = centroid(mesh.vertices)[:2]

    radii_2d = np.sqrt((mesh.vertices[:, 0] - center_xy[0])**2 +
                       (mesh.vertices[:, 1] - center_xy[1])**2)
    R = float(np.percentile(radii_2d, 90))
    z_min, z_max = float(bb_min[2]), float(bb_max[2])

    # Parametric stent topology
    N_circ = 8    # struts around circumference
    N_z    = 6    # rings along Z
    strut_r = R * 0.04  # estimate strut radius

    G = nx.Graph()
    node_id = 0
    node_grid = {}

    for iz in range(N_z):
        z = z_min + (z_max - z_min) * iz / (N_z - 1)
        for ic in range(N_circ):
            theta = 2 * np.pi * ic / N_circ + (iz % 2) * np.pi / N_circ
            x = center_xy[0] + R * np.cos(theta)
            y = center_xy[1] + R * np.sin(theta)
            G.add_node(node_id, pos=np.array([x, y, z]))
            node_grid[(iz, ic)] = node_id
            node_id += 1

    def add_beam(a, b):
        pa = G.nodes[a]['pos']
        pb = G.nodes[b]['pos']
        length = np.linalg.norm(pa - pb)
        G.add_edge(a, b, rest_length=length, radius=strut_r)

    for iz in range(N_z):
        for ic in range(N_circ):
            # circumferential ring
            ic_next = (ic + 1) % N_circ
            add_beam(node_grid[(iz, ic)], node_grid[(iz, ic_next)])
            # diagonal connecting rings (zigzag)
            if iz < N_z - 1:
                ic_diag = (ic + (iz % 2)) % N_circ
                add_beam(node_grid[(iz, ic)], node_grid[(iz + 1, ic_diag)])

    if verbose:
        print(f"[topology] Analytical fallback: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} beams")

    return BeamNetwork(G)


# ── diagnostic helper ──────────────────────────────────────────────────────────

def describe_network(network: BeamNetwork):
    """Print a summary of the beam network topology."""
    G = network.graph
    degrees = [G.degree(n) for n in G.nodes()]
    print(f"  Nodes : {network.num_nodes}")
    print(f"  Beams : {network.num_beams}")
    print(f"  Degree distribution: min={min(degrees)} max={max(degrees)} "
          f"mean={np.mean(degrees):.1f}")
    lengths = [G[u][v]['rest_length'] for u, v in G.edges()]
    print(f"  Beam lengths (mm): min={min(lengths):.3f} max={max(lengths):.3f} "
          f"mean={np.mean(lengths):.3f}")
