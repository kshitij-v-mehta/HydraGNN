#!/usr/bin/env python3
"""
aggregate_validation.py
-----------------------
Combines all per-rank validation_rankXXXXX.npz files produced by
validate_structures.py into a single summary and optional plots.

Usage:
    python aggregate_validation.py --out_dir ./validation_results
    python aggregate_validation.py --out_dir ./validation_results --plot
"""

import argparse
import glob
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True, help="Directory with validation_rank*.npz files")
    p.add_argument("--plot", action="store_true", help="Generate distribution plots (requires matplotlib)")
    p.add_argument("--plot_out", default="validation_summary.png", help="Plot output filename")
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.out_dir, "validation_rank*.npz")))
    if not files:
        print(f"No validation_rank*.npz files found in {args.out_dir}")
        return

    print(f"Aggregating {len(files)} rank files...")

    g_total      = 0
    g_invalid    = 0
    g_fail_e     = 0
    g_fail_f     = 0
    g_fail_d     = 0
    all_min_dists = []
    all_energies  = []

    for fpath in files:
        d = np.load(fpath, allow_pickle=False)
        g_total   += int(d["n_total"][0])
        g_invalid += int(d["n_invalid"][0])
        g_fail_e  += int(d["fail_energy"][0])
        g_fail_f  += int(d["fail_forces"][0])
        g_fail_d  += int(d["fail_mindist"][0])

        md = d["min_dist_all"]
        if len(md) > 0:
            # Only include structures where distance was actually computed (>= 0)
            valid_md = md[md >= 0]
            if len(valid_md) > 0:
                all_min_dists.append(valid_md)

        ep = d["energies_per_atom"]
        if len(ep) > 0:
            all_energies.append(ep)

    all_min_dists = np.concatenate(all_min_dists) if all_min_dists else np.array([])
    all_energies  = np.concatenate(all_energies)  if all_energies  else np.array([])

    valid_pct = 100.0 * (g_total - g_invalid) / max(g_total, 1)

    print("\n" + "="*60)
    print("GLOBAL VALIDATION SUMMARY")
    print("="*60)
    print(f"  Rank files aggregated      : {len(files):>15,}")
    print(f"  Total structures           : {g_total:>15,}")
    print(f"  Valid structures           : {g_total - g_invalid:>15,}  ({valid_pct:.4f}%)")
    print(f"  Invalid structures         : {g_invalid:>15,}  ({100-valid_pct:.4f}%)")
    print(f"    └─ Failed energy check   : {g_fail_e:>15,}")
    print(f"    └─ Failed force check    : {g_fail_f:>15,}")
    print(f"    └─ Failed min-dist check : {g_fail_d:>15,}")

    if len(all_min_dists) > 0:
        print(f"\n  Min-distance distribution (Å):")
        print(f"    min    = {all_min_dists.min():.4f}")
        print(f"    p01    = {np.percentile(all_min_dists, 1):.4f}")
        print(f"    p05    = {np.percentile(all_min_dists, 5):.4f}")
        print(f"    median = {np.median(all_min_dists):.4f}")
        print(f"    mean   = {all_min_dists.mean():.4f}")
        print(f"    max    = {all_min_dists.max():.4f}")

    if len(all_energies) > 0:
        print(f"\n  Formation energy/atom distribution (eV/atom):")
        print(f"    min    = {all_energies.min():.4f}")
        print(f"    p01    = {np.percentile(all_energies, 1):.4f}")
        print(f"    median = {np.median(all_energies):.4f}")
        print(f"    mean   = {all_energies.mean():.4f}")
        print(f"    p99    = {np.percentile(all_energies, 99):.4f}")
        print(f"    max    = {all_energies.max():.4f}")

    print("="*60)

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            if len(all_min_dists) > 0:
                axes[0].hist(all_min_dists, bins=100, color="steelblue", edgecolor="none")
                axes[0].axvline(0.5, color="red", linestyle="--", label="0.5 Å threshold")
                axes[0].set_xlabel("Min interatomic distance (Å)")
                axes[0].set_ylabel("Count")
                axes[0].set_title("Min Distance Distribution")
                axes[0].legend()

            if len(all_energies) > 0:
                lo, hi = np.percentile(all_energies, [0.5, 99.5])
                axes[1].hist(all_energies, bins=100, range=(lo, hi),
                             color="darkorange", edgecolor="none")
                axes[1].set_xlabel("Formation energy per atom (eV/atom)")
                axes[1].set_ylabel("Count")
                axes[1].set_title("Energy/Atom Distribution (valid structures)")

            plt.tight_layout()
            plt.savefig(args.plot_out, dpi=150)
            print(f"\nPlot saved to: {args.plot_out}")
        except ImportError:
            print("matplotlib not available — skipping plots")


if __name__ == "__main__":
    main()
