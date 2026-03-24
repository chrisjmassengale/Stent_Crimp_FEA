"""
simulate.py — CLI entry point for stent crimping/deployment simulation.

Usage:
    python simulate.py --input stent.stl \\
                       --crimp-diameter 6.0 \\
                       --deployed-diameter 28.0 \\
                       --output-dir ./frames

Options:
    --input            Path to input STL file
    --crimp-diameter   Inner diameter of delivery tube (mm)
    --deployed-diameter Stent natural expanded diameter (mm)
    --output-dir       Directory to write output STL frames (default: ./frames)
    --n-crimp-steps    Number of crimping load steps (default: 50)
    --n-deploy-steps   Number of deployment load steps (default: 50)
    --axis             Retraction axis: x, y, or z (default: z)
    --verbose          Print progress (default: True)
    --test-material    Run material model self-test and exit
    --test-topology    Run topology extraction only and report, then exit
"""

import argparse
import sys
import os
import numpy as np

from topology import load_and_extract, describe_network
from material import plot_stress_strain_curve, NITINOL
from solver import simulate
from deform import export_frames, check_mesh_quality


def parse_args():
    p = argparse.ArgumentParser(
        description="Stent crimping & deployment FEA simulator (visual)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("--input", "-i", type=str, default=None,
                   help="Path to input STL file")
    p.add_argument("--crimp-diameter", "-c", type=float, default=6.0,
                   help="Inner diameter of delivery tube in mm (default: 6.0)")
    p.add_argument("--deployed-diameter", "-d", type=float, default=28.0,
                   help="Stent natural expanded diameter in mm (default: 28.0)")
    p.add_argument("--output-dir", "-o", type=str, default="./frames",
                   help="Output directory for STL frames (default: ./frames)")
    p.add_argument("--n-crimp-steps", type=int, default=50,
                   help="Number of crimping steps (default: 50)")
    p.add_argument("--n-deploy-steps", type=int, default=50,
                   help="Number of deployment steps (default: 50)")
    p.add_argument("--axis", choices=["x", "y", "z"], default="z",
                   help="Tube retraction axis (default: z)")
    p.add_argument("--no-verbose", action="store_true",
                   help="Suppress progress output")
    p.add_argument("--test-material", action="store_true",
                   help="Run material model self-test and exit")
    p.add_argument("--test-topology", action="store_true",
                   help="Run topology extraction only and exit")
    p.add_argument("--strategy", choices=["auto", "rbf", "idw"], default="auto",
                   help="Mesh deformation interpolation strategy (default: auto)")
    p.add_argument("--transition-length", type=float, default=0.45,
                   help="Transition zone as fraction of stent height (default: 0.45)")
    p.add_argument("--snap-speed", type=float, default=3.0,
                   help="Snap-back exponent: 1-(1-t)^n, higher=snappier (default: 3.0)")
    p.add_argument("--crown-dwell", type=float, default=0.60,
                   help="Crown dwell fraction 0-1: how long crowns stay crimped (default: 0.60)")
    p.add_argument("--expansion-exponent", type=float, default=0.6,
                   help="Global expansion curve exponent: lower=more constrained early (default: 0.6)")
    p.add_argument("--tine-flare", type=float, default=1.15,
                   help="Crown tine flare factor: tines expand to this * deployed_r (default: 1.15)")
    return p.parse_args()


def run_material_test():
    print("=" * 60)
    print("Nitinol superelastic material model self-test")
    print("=" * 60)
    plot_stress_strain_curve(NITINOL)

    # Numerical checks
    from material import stress_from_strain, tangent_modulus
    print("\nSpot checks:")
    checks = [
        (0.005, 0.005, "elastic loading", 0.005 * NITINOL.E_A),
        (0.04,  0.04,  "upper plateau",   None),
        (0.04,  0.08,  "unloading",       None),
        (0.001, 0.08,  "full unload",     None),
    ]
    all_ok = True
    for eps, eps_max, label, expected in checks:
        sig = stress_from_strain(eps, eps_max) / 1e6
        Et  = tangent_modulus(eps, eps_max) / 1e9
        ok  = True
        if expected is not None:
            ok = abs(stress_from_strain(eps, eps_max) - expected) < 0.01 * abs(expected) + 1
        print(f"  [{label:20s}] eps={eps:.3f} eps_max={eps_max:.3f} "
              f"  sig={sig:6.1f} MPa  Et={Et:.1f} GPa  {'OK' if ok else 'FAIL'}")
        if not ok:
            all_ok = False
    print(f"\nMaterial test: {'PASSED' if all_ok else 'FAILED'}")
    return 0 if all_ok else 1


def run_topology_test(stl_path):
    print("=" * 60)
    print(f"Topology extraction test: {stl_path}")
    print("=" * 60)
    try:
        mesh, network = load_and_extract(stl_path, verbose=True, deployed_diameter_mm=None)
        print("\nMesh quality:")
        q = check_mesh_quality(mesh)
        for k, v in q.items():
            print(f"  {k}: {v}")
        print("\nBeam network:")
        describe_network(network)
        print("\nTopology test: PASSED")
        return 0
    except Exception as e:
        print(f"\nTopology test: FAILED — {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    args = parse_args()
    verbose = not args.no_verbose

    # Override deformation strategy if specified
    if args.strategy != "auto":
        import deform
        deform._STRATEGY = args.strategy

    # ── self-tests ─────────────────────────────────────────────────────────────
    if args.test_material:
        sys.exit(run_material_test())

    if args.test_topology:
        if not args.input:
            print("Error: --input is required for --test-topology", file=sys.stderr)
            sys.exit(1)
        sys.exit(run_topology_test(args.input))

    # ── main simulation pipeline ──────────────────────────────────────────────
    if not args.input:
        print("Error: --input STL file is required.\n", file=sys.stderr)
        print("Run 'python simulate.py --help' for usage.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input!r}", file=sys.stderr)
        sys.exit(1)

    if args.crimp_diameter >= args.deployed_diameter:
        print("Error: crimp diameter must be smaller than deployed diameter.",
              file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Stent Crimping/Deployment Simulator")
    print("=" * 60)
    print(f"  Input STL        : {args.input}")
    print(f"  Crimp diameter   : {args.crimp_diameter} mm")
    print(f"  Deployed diameter: {args.deployed_diameter} mm")
    print(f"  Output directory : {args.output_dir}")
    print(f"  Crimping steps   : {args.n_crimp_steps}")
    print(f"  Deployment steps : {args.n_deploy_steps}")
    print(f"  Transition length: {args.transition_length}")
    print(f"  Snap speed       : {args.snap_speed}")
    print(f"  Crown dwell      : {args.crown_dwell}")
    print(f"  Expansion exp    : {args.expansion_exponent}")
    print()

    # Step 1: Topology extraction
    print("[1/4] Extracting beam network from STL...")
    try:
        mesh, network = load_and_extract(args.input, verbose=verbose,
                                         deployed_diameter_mm=args.deployed_diameter)
    except Exception as e:
        print(f"Fatal: topology extraction failed — {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    describe_network(network)
    print()

    # Step 2: Run FEA simulation
    print("[2/4] Running FEA simulation...")
    try:
        frames, meta = simulate(
            network,
            crimp_diameter_mm    = args.crimp_diameter,
            deployed_diameter_mm = args.deployed_diameter,
            n_crimp_steps        = args.n_crimp_steps,
            n_deploy_steps       = args.n_deploy_steps,
            verbose              = verbose,
        )
    except Exception as e:
        print(f"Fatal: simulation failed — {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    total_frames = len(frames)
    print(f"  Generated {total_frames} frames")
    print()

    # Step 3: Reorder and resample to exactly 100 frames.
    # frames[0..n_crimp-1]     = crimping   (deployed → fully crimped)
    # frames[n_crimp..total-1] = deployment (crimped  → fully deployed)
    # Output spec: frame_000 = fully crimped, frame_099 = fully deployed.
    # We output: [fully crimped state] + [deployment animation] → resample to 100.
    print("[3/4] Resampling to 100 output frames...")
    n_crimp = args.n_crimp_steps
    fully_crimped_frame = frames[n_crimp - 1]     # last crimp step
    fully_crimped_meta  = meta[n_crimp - 1]
    deploy_sequence     = frames[n_crimp:]
    deploy_meta_seq     = meta[n_crimp:]

    # Build ordered sequence: hold crimped briefly then show deployment
    ordered      = [fully_crimped_frame] + deploy_sequence
    ordered_meta = [fully_crimped_meta]  + deploy_meta_seq

    indices = np.linspace(0, len(ordered) - 1, 100).astype(int)
    frames_out = [ordered[i]      for i in indices]
    meta_out   = [ordered_meta[i] for i in indices]

    # Step 4: Export STL files
    print("[4/4] Exporting STL frames...")
    try:
        paths = export_frames(mesh, network, frames_out, meta_out,
                              args.output_dir, verbose=verbose,
                              transition_frac=args.transition_length,
                              snap_speed=args.snap_speed,
                              crown_dwell=args.crown_dwell,
                              expansion_exponent=args.expansion_exponent,
                              tine_flare=args.tine_flare)
    except Exception as e:
        print(f"Fatal: frame export failed — {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()
    print("=" * 60)
    print(f"Done! {len(paths)} frames written to: {args.output_dir}")
    print(f"  frame_000.stl = fully crimped")
    print(f"  frame_099.stl = fully deployed")
    print("=" * 60)

    # Launch the interactive viewer
    viewer_path = os.path.join(os.path.dirname(__file__), "viewer.py")
    if os.path.isfile(viewer_path):
        print("\nLaunching viewer...")
        try:
            import subprocess
            subprocess.Popen(
                [sys.executable, viewer_path, args.output_dir],
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
        except Exception as e:
            print(f"Could not launch viewer: {e}")
            print(f"Run manually:  python viewer.py {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
