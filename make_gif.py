"""
Render all STL frames to a deployment animation GIF.
Side view (XZ plane), wireframe + filled, auto-scaling.

Can be run standalone or imported and called via generate_gif().
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from pathlib import Path
from PIL import Image
import io


def generate_gif(frames_dir="./frames",
                 output_path="deployment_animation.gif",
                 fps=15,
                 max_frames=20,
                 dpi=60,
                 verbose=True):
    """
    Render all frame_*.stl files in *frames_dir* to an animated GIF.

    Parameters
    ----------
    frames_dir  : str or Path — directory containing frame_NNN.stl files
    output_path : str or Path — output GIF path
    fps         : int         — frames per second
    verbose     : bool        — print progress
    """
    frames_dir  = Path(frames_dir)
    output_path = Path(output_path)

    all_stl_files = sorted(frames_dir.glob("frame_*.stl"))
    if not all_stl_files:
        raise FileNotFoundError(f"No frame_*.stl files found in {frames_dir}")

    # Subsample to at most max_frames
    if len(all_stl_files) > max_frames:
        indices = np.linspace(0, len(all_stl_files) - 1, max_frames, dtype=int)
        stl_files = [all_stl_files[i] for i in indices]
    else:
        stl_files = all_stl_files

    if verbose:
        print(f"[gif] Found {len(all_stl_files)} frames, using {len(stl_files)}")

    # Global bounds from first and last frame
    bounds_list = []
    for sf in [stl_files[0], stl_files[-1]]:
        m = trimesh.load(str(sf), process=False, force='mesh')
        bounds_list.append(m.bounds)
    all_bounds = np.array(bounds_list)
    x_min = all_bounds[:, 0, 0].min()
    x_max = all_bounds[:, 1, 0].max()
    z_min = all_bounds[:, 0, 2].min()
    z_max = all_bounds[:, 1, 2].max()
    pad   = max(x_max - x_min, z_max - z_min) * 0.05
    x_min -= pad; x_max += pad
    z_min -= pad; z_max += pad

    images = []
    for i, sf in enumerate(stl_files):
        m = trimesh.load(str(sf), process=False, force='mesh')
        v = m.vertices
        f = m.faces

        fig, ax = plt.subplots(1, 1, figsize=(3, 6), dpi=dpi)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        ax.set_aspect('equal')
        ax.set_facecolor('#2d2d4e')
        fig.patch.set_facecolor('#2d2d4e')
        ax.axis('off')

        # Project faces onto XZ plane, sort back-to-front by Y depth
        centroids_y = v[f].mean(axis=1)[:, 1]
        sort_order  = np.argsort(centroids_y)

        tris_xz   = v[f[sort_order]][:, :, [0, 2]]
        normals_y = m.face_normals[sort_order, 1]

        shade  = 0.35 + 0.65 * np.clip(np.abs(normals_y), 0, 1)
        colors = np.column_stack(
            [shade * 0.85, shade * 0.85, shade * 0.88, np.ones(len(shade))])

        poly = PolyCollection(tris_xz, facecolors=colors,
                              edgecolors='none', linewidths=0)
        ax.add_collection(poly)

        ax.text(0.02, 0.98, f'Frame {i:03d}', transform=ax.transAxes,
                fontsize=8, color='white', va='top', family='monospace')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).copy())
        buf.close()

        if verbose and (i == 0 or i == len(stl_files) - 1 or (i + 1) % 5 == 0):
            print(f"[gif]   Rendered frame {i + 1}/{len(stl_files)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        str(output_path),
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    if verbose:
        print(f"[gif] Saved: {output_path}  ({len(images)} frames @ {fps} fps)")


if __name__ == "__main__":
    generate_gif(
        frames_dir="./frames",
        output_path="deployment_animation.gif",
        fps=15,
        max_frames=20,
        dpi=60,
        verbose=True,
    )
