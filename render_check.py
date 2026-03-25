"""
render_check.py — Render specific frame STL files to PNG for inspection.
Usage: python render_check.py [frames_dir] [frame_indices...]
       python render_check.py ./frames 0 30 40 50 55 60 70 99
"""

import sys
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from pathlib import Path


def render_frame(stl_path, ax, x_min, x_max, z_min, z_max):
    m = trimesh.load(str(stl_path), process=False, force='mesh')
    v = m.vertices
    f = m.faces

    # Project onto XZ plane, sort back-to-front by Y
    centroids_y = v[f].mean(axis=1)[:, 1]
    sort_order  = np.argsort(centroids_y)
    tris_xz     = v[f[sort_order]][:, :, [0, 2]]
    normals_y   = m.face_normals[sort_order, 1]

    shade  = 0.35 + 0.65 * np.clip(np.abs(normals_y), 0, 1)
    colors = np.column_stack(
        [shade * 0.85, shade * 0.85, shade * 0.88, np.ones(len(shade))])
    poly = PolyCollection(tris_xz, facecolors=colors, edgecolors='none', linewidths=0)
    ax.add_collection(poly)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_aspect('equal')


def main():
    frames_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./frames')
    if len(sys.argv) > 2:
        indices = [int(x) for x in sys.argv[2:]]
    else:
        indices = [0, 30, 40, 50, 55, 60, 70, 99]

    # Find available frames
    stl_files = {}
    for p in frames_dir.glob('frame_*.stl'):
        n = int(p.stem.split('_')[1])
        stl_files[n] = p

    # Global bounds from frame 0 and 99
    bounds = []
    for n in [min(stl_files), max(stl_files)]:
        m = trimesh.load(str(stl_files[n]), process=False, force='mesh')
        bounds.append(m.bounds)
    b = np.array(bounds)
    pad = 2.0
    x_min = b[:, 0, 0].min() - pad;  x_max = b[:, 1, 0].max() + pad
    z_min = b[:, 0, 2].min() - pad;  z_max = b[:, 1, 2].max() + pad

    out_dir = Path(__file__).parent
    for n in indices:
        if n not in stl_files:
            print(f"  [skip] frame_{n:03d}.stl not found")
            continue
        fig, ax = plt.subplots(1, 1, figsize=(4, 8), dpi=80)
        ax.set_facecolor('#2d2d4e')
        fig.patch.set_facecolor('#2d2d4e')
        ax.axis('off')
        render_frame(stl_files[n], ax, x_min, x_max, z_min, z_max)
        ax.set_title(f'frame {n:03d}', color='white', fontsize=10, pad=4)
        out = out_dir / f'debug_frame{n:03d}.png'
        fig.savefig(str(out), bbox_inches='tight', pad_inches=0.05,
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved {out}")


if __name__ == '__main__':
    main()
