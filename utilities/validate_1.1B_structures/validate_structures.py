#!/usr/bin/env python3
"""
validate_structures.py
----------------------
MPI-parallel validation of atomistic structures stored in ADIOS2 BP5 files
produced by inference_fused_write_adios.py.

Layout:
  - BP5 files are packed one-per-tar on Lustre
  - Each rank extracts its assigned tar files to /tmp (node-local),
    validates the BP5 inside, then cleans up before moving on
  - One ADIOS step  = one structure
  - One tar file    = one BP5 directory = 120,000 structures
  - 9,300 tar files distributed round-robin across all MPI ranks

Checks (fast-to-slow gate):
  1. formation_energy finite + within [energy_lo, energy_hi] eV
  2. force magnitudes finite and below max_force eV/A
  3. N >= 2  (skip distance check for monomers)
  4. min interatomic distance >= min_dist A  (cKDTree, no PBC)

Output per rank:
  {out_dir}/validation_rank{RANK:05d}.npz with aggregate stats and arrays.

Usage (test, no MPI):
  python validate_structures.py --tar_dir /path/to/tars --out_dir ./results --max_structs 100

Usage (MPI):
  srun -n <N> python validate_structures.py \
      --tar_dir /lustre/orion/lrn070/.../tars \
      --out_dir /lustre/orion/lrn070/.../validation_results
"""

import argparse
import glob
import os
import shutil
import sys
import tarfile
import tempfile
import time

import numpy as np

try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    SIZE = 1

try:
    import adios2.bindings as adios2
except ImportError:
    print("ERROR: adios2 not available. Load the adios2 module before running.", file=sys.stderr)
    sys.exit(1)

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MPI-parallel ADIOS2 structure validator")
    p.add_argument("--tar_dir",    required=True,
                   help="Directory on Lustre containing all *.tar files")
    p.add_argument("--tar_pattern", default="*.tar",
                   help="Glob pattern for tar files (default: *.tar)")
    p.add_argument("--tmp_dir",    default="/tmp",
                   help="Node-local directory for BP5 extraction (default: /tmp)")
    p.add_argument("--out_dir",    required=True,
                   help="Output directory for per-rank .npz result files (on Lustre)")
    p.add_argument("--min_dist",   type=float, default=0.5,
                   help="Min interatomic distance in A (default: 0.5)")
    p.add_argument("--max_force",  type=float, default=50.0,
                   help="Max force magnitude eV/A (default: 50.0)")
    p.add_argument("--energy_lo",  type=float, default=-20.0,
                   help="Min allowed formation energy eV (default: -20.0)")
    p.add_argument("--energy_hi",  type=float, default=20.0,
                   help="Max allowed formation energy eV (default: 20.0)")
    p.add_argument("--max_structs", type=int, default=0,
                   help="Max structures per file (0=all; useful for testing)")
    p.add_argument("--no_mindist", action="store_true",
                   help="Skip cKDTree min-distance check (energy+force only)")
    p.add_argument("--verbose",    action="store_true",
                   help="Print per-file progress")
    return p.parse_args()


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def get_tar_files(tar_dir, pattern):
    hits = sorted(glob.glob(os.path.join(tar_dir, pattern)))
    return hits


def assign_files_to_rank(all_files, rank, size):
    """Round-robin: rank owns files at indices rank, rank+size, rank+2*size, ..."""
    return [f for i, f in enumerate(all_files) if i % size == rank]


def extract_tar(tar_path, extract_dir):
    """
    Extract a single tar file into extract_dir.
    Returns the path to the extracted BP5 directory (first .bp entry found),
    or None if nothing matching is found.
    """
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(path=extract_dir)

    # Find the .bp directory that was extracted
    bp_dirs = glob.glob(os.path.join(extract_dir, "*.bp"))
    # Also check one level deep in case tar has a subdirectory
    if not bp_dirs:
        bp_dirs = glob.glob(os.path.join(extract_dir, "*", "*.bp"))

    if not bp_dirs:
        return None
    return bp_dirs[0]


def cleanup_extract(extract_dir):
    """Remove everything in extract_dir (the per-file temp working directory)."""
    shutil.rmtree(extract_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Per-file validation
# ---------------------------------------------------------------------------

def validate_bp_file(bp_path, args):
    """
    Step through all ADIOS steps in bp_path and validate each structure.
    Returns a dict of aggregate stats + per-structure arrays, or None on error.
    """
    min_dist_thresh = args.min_dist
    max_force       = args.max_force
    energy_lo       = args.energy_lo
    energy_hi       = args.energy_hi
    do_mindist      = (not args.no_mindist) and (cKDTree is not None)
    max_structs     = args.max_structs  # 0 = unlimited

    n_total      = 0
    n_invalid    = 0
    fail_energy  = 0
    fail_forces  = 0
    fail_mindist = 0
    bad_indices  = []
    min_dists    = []
    energies_pa  = []

    a  = adios2.ADIOS()
    io = a.DeclareIO(f"reader_{RANK}_{os.getpid()}")
    io.SetEngine("BP5")

    try:
        reader = io.Open(bp_path, adios2.Mode.Read)
    except Exception as e:
        print(f"[rank {RANK}] ERROR opening {bp_path}: {e}", file=sys.stderr)
        return None

    step_idx = 0
    while True:
        status = reader.BeginStep()
        if status != adios2.StepStatus.OK:
            break

        var_x  = io.InquireVariable("coordinates_x")
        var_y  = io.InquireVariable("coordinates_y")
        var_z  = io.InquireVariable("coordinates_z")
        var_fx = io.InquireVariable("forces_x")
        var_fy = io.InquireVariable("forces_y")
        var_fz = io.InquireVariable("forces_z")
        var_e  = io.InquireVariable("formation_energy")

        if var_x is None:
            reader.EndStep()
            step_idx += 1
            continue

        N = var_x.Shape()[0]

        cx = np.empty(N, dtype=np.float64)
        cy = np.empty(N, dtype=np.float64)
        cz = np.empty(N, dtype=np.float64)
        fx = np.empty(N, dtype=np.float64)
        fy = np.empty(N, dtype=np.float64)
        fz = np.empty(N, dtype=np.float64)
        en = np.empty(1, dtype=np.float64)

        reader.Get(var_x,  cx)
        reader.Get(var_y,  cy)
        reader.Get(var_z,  cz)
        reader.Get(var_fx, fx)
        reader.Get(var_fy, fy)
        reader.Get(var_fz, fz)
        reader.Get(var_e,  en)

        reader.EndStep()  # deferred Gets complete here

        n_total += 1
        is_bad   = False
        min_d    = -1.0

        # Check 1: energy
        e_val = en[0]
        if not np.isfinite(e_val) or e_val < energy_lo or e_val > energy_hi:
            fail_energy += 1
            is_bad = True

        # Check 2: forces
        if not is_bad:
            f_mag_sq = fx**2 + fy**2 + fz**2
            if not np.all(np.isfinite(f_mag_sq)) or f_mag_sq.max() > max_force**2:
                fail_forces += 1
                is_bad = True

        # Check 3+4: min interatomic distance
        if not is_bad and do_mindist and N >= 2:
            coords = np.stack([cx, cy, cz], axis=1)
            tree   = cKDTree(coords)
            dists, _ = tree.query(coords, k=2)
            min_d  = float(dists[:, 1].min())
            if min_d < min_dist_thresh:
                fail_mindist += 1
                is_bad = True

        min_dists.append(min_d)

        if is_bad:
            n_invalid += 1
            bad_indices.append(step_idx)
        else:
            energies_pa.append(e_val / N)

        step_idx += 1
        if max_structs > 0 and step_idx >= max_structs:
            break

    reader.Close()

    return {
        "n_total"          : n_total,
        "n_invalid"        : n_invalid,
        "fail_energy"      : fail_energy,
        "fail_forces"      : fail_forces,
        "fail_mindist"     : fail_mindist,
        "bad_indices"      : np.array(bad_indices, dtype=np.int64),
        "min_dist_all"     : np.array(min_dists,   dtype=np.float32),
        "energies_per_atom": np.array(energies_pa, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Rank 0 discovers tar files, broadcasts list to all ranks
    if RANK == 0:
        all_tars = get_tar_files(args.tar_dir, args.tar_pattern)
        if not all_tars:
            print(f"ERROR: No files matching '{args.tar_pattern}' in '{args.tar_dir}'",
                  file=sys.stderr)
            if COMM:
                COMM.Abort(1)
            sys.exit(1)
        print(f"[rank 0] Found {len(all_tars)} tar files. "
              f"Distributing across {SIZE} MPI ranks "
              f"(~{len(all_tars)//SIZE} files/rank).", flush=True)
    else:
        all_tars = None

    if COMM:
        all_tars = COMM.bcast(all_tars, root=0)

    my_tars = assign_files_to_rank(all_tars, RANK, SIZE)

    if RANK == 0:
        os.makedirs(args.out_dir, exist_ok=True)
    if COMM:
        COMM.Barrier()

    # ---- Per-rank extraction + validation loop ----
    rank_n_total      = 0
    rank_n_invalid    = 0
    rank_fail_energy  = 0
    rank_fail_forces  = 0
    rank_fail_mindist = 0
    rank_bad_indices  = []
    rank_min_dists    = []
    rank_energies_pa  = []

    for file_idx, tar_path in enumerate(my_tars):
        tar_name = os.path.basename(tar_path)

        # Each rank uses a unique subdir in /tmp to avoid collisions
        # between ranks sharing the same node
        extract_dir = os.path.join(args.tmp_dir, f"validate_rank{RANK}_{file_idx}")
        os.makedirs(extract_dir, exist_ok=True)

        # --- Extract ---
        t_extract = time.perf_counter()
        try:
            bp_path = extract_tar(tar_path, extract_dir)
        except Exception as e:
            print(f"[rank {RANK}] ERROR extracting {tar_name}: {e}", file=sys.stderr)
            cleanup_extract(extract_dir)
            continue

        if bp_path is None:
            print(f"[rank {RANK}] WARNING: no .bp directory found in {tar_name}", file=sys.stderr)
            cleanup_extract(extract_dir)
            continue

        t_extract = time.perf_counter() - t_extract

        # --- Validate ---
        t_validate = time.perf_counter()
        result = validate_bp_file(bp_path, args)
        t_validate = time.perf_counter() - t_validate

        # --- Cleanup /tmp immediately to free node-local space ---
        cleanup_extract(extract_dir)

        if result is None:
            continue

        rank_n_total      += result["n_total"]
        rank_n_invalid    += result["n_invalid"]
        rank_fail_energy  += result["fail_energy"]
        rank_fail_forces  += result["fail_forces"]
        rank_fail_mindist += result["fail_mindist"]
        rank_bad_indices.append(result["bad_indices"])
        rank_min_dists.append(result["min_dist_all"])
        rank_energies_pa.append(result["energies_per_atom"])

        if args.verbose:
            pct = 100.0 * result["n_invalid"] / max(result["n_total"], 1)
            print(f"[rank {RANK:05d}] [{file_idx+1:4d}/{len(my_tars)}] {tar_name} "
                  f"structs={result['n_total']} invalid={result['n_invalid']} ({pct:.2f}%) "
                  f"extract={t_extract:.1f}s validate={t_validate:.1f}s", flush=True)

    # ---- Save per-rank results to Lustre ----
    out_path = os.path.join(args.out_dir, f"validation_rank{RANK:05d}.npz")
    np.savez_compressed(
        out_path,
        n_total       = np.array([rank_n_total],      dtype=np.int64),
        n_invalid     = np.array([rank_n_invalid],     dtype=np.int64),
        fail_energy   = np.array([rank_fail_energy],   dtype=np.int64),
        fail_forces   = np.array([rank_fail_forces],   dtype=np.int64),
        fail_mindist  = np.array([rank_fail_mindist],  dtype=np.int64),
        bad_indices   = np.concatenate(rank_bad_indices)  if rank_bad_indices  else np.array([], dtype=np.int64),
        min_dist_all  = np.concatenate(rank_min_dists)    if rank_min_dists    else np.array([], dtype=np.float32),
        energies_per_atom = np.concatenate(rank_energies_pa) if rank_energies_pa else np.array([], dtype=np.float32),
    )

    # ---- Global reduction for summary ----
    if COMM:
        COMM.Barrier()
        g_total   = COMM.reduce(rank_n_total,      op=MPI.SUM, root=0)
        g_invalid = COMM.reduce(rank_n_invalid,    op=MPI.SUM, root=0)
        g_fail_e  = COMM.reduce(rank_fail_energy,  op=MPI.SUM, root=0)
        g_fail_f  = COMM.reduce(rank_fail_forces,  op=MPI.SUM, root=0)
        g_fail_d  = COMM.reduce(rank_fail_mindist, op=MPI.SUM, root=0)
    else:
        g_total   = rank_n_total
        g_invalid = rank_n_invalid
        g_fail_e  = rank_fail_energy
        g_fail_f  = rank_fail_forces
        g_fail_d  = rank_fail_mindist

    if RANK == 0:
        valid_pct = 100.0 * (g_total - g_invalid) / max(g_total, 1)
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"  Total structures processed : {g_total:>15,}")
        print(f"  Valid structures           : {g_total - g_invalid:>15,}  ({valid_pct:.3f}%)")
        print(f"  Invalid structures         : {g_invalid:>15,}  ({100-valid_pct:.3f}%)")
        print(f"    └─ Failed energy check   : {g_fail_e:>15,}")
        print(f"    └─ Failed force check    : {g_fail_f:>15,}")
        print(f"    └─ Failed min-dist check : {g_fail_d:>15,}")
        print(f"  Per-rank .npz files in     : {args.out_dir}")
        print("="*60)


if __name__ == "__main__":
    main()
