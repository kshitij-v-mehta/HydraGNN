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

Validation logic lives in structure_checks.py (must be in the same directory).

Checks applied in order (fast-to-slow gate):
  Stage 1 — Physical validity:
    1a. formation_energy finite + within [energy_lo, energy_hi] eV
    1b. force magnitudes finite and below max_force eV/A
    1c+1d. min interatomic distance >= min_dist A  (cKDTree, no PBC)

  Stage 2 — Synthesizability (Stage 1 survivors only):
    2a. All elements in synthesizable whitelist (Z 1..83, no Tc/Pm)
    2b. No isolated atoms
    2c. No atom exceeds its maximum valence
    2d. No radicals (odd bond count on C/N/O/H)

Output per rank:
  {out_dir}/validation_rank{RANK:05d}.npz

Usage (test, no MPI):
  python validate_structures.py --tar_dir /path/to/tars --out_dir ./results --max_structs 100

Usage (MPI):
  srun -n <N> python validate_structures.py \\
      --tar_dir /lustre/orion/lrn070/.../tars \\
      --out_dir /lustre/orion/lrn070/.../validation_results

Flags:
  --tar_dir       Directory containing all *.tar files
  --tar_pattern   Glob pattern for tar files (default: *.tar)
  --tmp_dir       Node-local extraction directory (default: /tmp)
  --out_dir       Output directory for per-rank .npz result files
  --min_dist      Min interatomic distance in A (default: 0.5)
  --max_force     Max force magnitude eV/A (default: 50.0)
  --energy_lo     Min allowed formation energy eV (default: -20.0)
  --energy_hi     Max allowed formation energy eV (default: 20.0)
  --max_structs   Max structures per file, 0=all (for testing)
  --no_mindist    Skip min-distance check
  --no_synth      Skip Stage 2 synthesizability checks
  --verbose       Print per-file progress
"""

import argparse
import glob
import os
import shutil
import sys
import tarfile
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
    print("ERROR: adios2 not available.", file=sys.stderr)
    sys.exit(1)

from structure_checks import check_stage1, check_stage2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MPI-parallel ADIOS2 structure validator")
    p.add_argument("--tar_dir",     required=True)
    p.add_argument("--tar_pattern", default="*.tar")
    p.add_argument("--tmp_dir",     default="/tmp")
    p.add_argument("--out_dir",     required=True)
    p.add_argument("--min_dist",    type=float, default=0.5)
    p.add_argument("--max_force",   type=float, default=50.0)
    p.add_argument("--energy_lo",   type=float, default=-20.0)
    p.add_argument("--energy_hi",   type=float, default=20.0)
    p.add_argument("--max_structs", type=int,   default=0)
    p.add_argument("--no_mindist",  action="store_true")
    p.add_argument("--no_synth",    action="store_true")
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def get_tar_files(tar_dir, pattern):
    return sorted(glob.glob(os.path.join(tar_dir, pattern)))


def assign_files_to_rank(all_files, rank, size):
    return [f for i, f in enumerate(all_files) if i % size == rank]


def extract_tar(tar_path, extract_dir):
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(path=extract_dir)
    bp_dirs = glob.glob(os.path.join(extract_dir, "*.bp"))
    if not bp_dirs:
        bp_dirs = glob.glob(os.path.join(extract_dir, "*", "*.bp"))
    return bp_dirs[0] if bp_dirs else None


def cleanup_extract(extract_dir):
    shutil.rmtree(extract_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Per-file validation
# ---------------------------------------------------------------------------

def validate_bp_file(bp_path, args):
    """
    Step through all ADIOS steps in bp_path and validate each structure.
    Returns a dict of aggregate stats + per-structure arrays, or None on error.
    """
    do_mindist  = not args.no_mindist
    do_synth    = not args.no_synth
    max_structs = args.max_structs

    # Stage 1 counters
    n_total      = 0
    n_invalid    = 0
    fail_energy  = 0
    fail_forces  = 0
    fail_mindist = 0

    # Stage 2 counters
    n_synth_checked = 0
    n_synth_fail    = 0
    fail_element    = 0
    fail_isolated   = 0
    fail_valence    = 0
    fail_radical    = 0
    n_synthesizable = 0

    bad_indices       = []
    non_synth_indices = []
    min_dists         = []
    energies_pa       = []
    synth_energies_pa = []

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
        var_at = io.InquireVariable("atom_types")

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
        at = np.empty(N, dtype=np.int32)

        reader.Get(var_x,  cx)
        reader.Get(var_y,  cy)
        reader.Get(var_z,  cz)
        reader.Get(var_fx, fx)
        reader.Get(var_fy, fy)
        reader.Get(var_fz, fz)
        reader.Get(var_e,  en)
        if var_at is not None:
            reader.Get(var_at, at)

        reader.EndStep()  # deferred Gets complete here

        n_total += 1

        # ----------------------------------------------------------------
        # Stage 1: physical validity
        # ----------------------------------------------------------------
        s1_ok, s1_reason, min_d = check_stage1(
            cx, cy, cz, fx, fy, fz, en[0], N,
            energy_lo  = args.energy_lo,
            energy_hi  = args.energy_hi,
            max_force  = args.max_force,
            min_dist   = args.min_dist,
            do_mindist = do_mindist,
        )

        min_dists.append(min_d)

        if not s1_ok:
            n_invalid += 1
            bad_indices.append(step_idx)
            if "energy"   in s1_reason: fail_energy  += 1
            elif "force"  in s1_reason: fail_forces  += 1
            elif "overlap" in s1_reason: fail_mindist += 1
            step_idx += 1
            if max_structs > 0 and step_idx >= max_structs:
                break
            continue

        energies_pa.append(en[0] / N)

        # ----------------------------------------------------------------
        # Stage 2: synthesizability
        # ----------------------------------------------------------------
        if not do_synth or var_at is None:
            n_synthesizable += 1
            synth_energies_pa.append(en[0] / N)
            step_idx += 1
            if max_structs > 0 and step_idx >= max_structs:
                break
            continue

        n_synth_checked += 1
        s2_ok, s2_reason = check_stage2(cx, cy, cz, at, N)

        if not s2_ok:
            n_synth_fail += 1
            non_synth_indices.append(step_idx)
            if "element"  in s2_reason: fail_element  += 1
            elif "isolated" in s2_reason: fail_isolated += 1
            elif "valence"  in s2_reason: fail_valence  += 1
            elif "radical"  in s2_reason: fail_radical  += 1
        else:
            n_synthesizable += 1
            synth_energies_pa.append(en[0] / N)

        step_idx += 1
        if max_structs > 0 and step_idx >= max_structs:
            break

    reader.Close()

    return {
        "n_total"           : n_total,
        "n_invalid"         : n_invalid,
        "fail_energy"       : fail_energy,
        "fail_forces"       : fail_forces,
        "fail_mindist"      : fail_mindist,
        "bad_indices"       : np.array(bad_indices,       dtype=np.int64),
        "min_dist_all"      : np.array(min_dists,         dtype=np.float32),
        "energies_per_atom" : np.array(energies_pa,       dtype=np.float32),
        "n_synth_checked"   : n_synth_checked,
        "n_synth_fail"      : n_synth_fail,
        "fail_element"      : fail_element,
        "fail_isolated"     : fail_isolated,
        "fail_valence"      : fail_valence,
        "fail_radical"      : fail_radical,
        "n_synthesizable"   : n_synthesizable,
        "non_synth_indices" : np.array(non_synth_indices, dtype=np.int64),
        "synth_energies_pa" : np.array(synth_energies_pa, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

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

    # Rank-local accumulators
    rank = {
        "n_total": 0, "n_invalid": 0,
        "fail_energy": 0, "fail_forces": 0, "fail_mindist": 0,
        "n_synth_checked": 0, "n_synth_fail": 0,
        "fail_element": 0, "fail_isolated": 0,
        "fail_valence": 0, "fail_radical": 0,
        "n_synthesizable": 0,
    }
    rank_bad_indices       = []
    rank_non_synth_indices = []
    rank_min_dists         = []
    rank_energies_pa       = []
    rank_synth_energies    = []

    for file_idx, tar_path in enumerate(my_tars):
        tar_name    = os.path.basename(tar_path)
        extract_dir = os.path.join(args.tmp_dir, f"validate_rank{RANK}_{file_idx}")
        os.makedirs(extract_dir, exist_ok=True)

        t_extract = time.perf_counter()
        try:
            bp_path = extract_tar(tar_path, extract_dir)
        except Exception as e:
            print(f"[rank {RANK}] ERROR extracting {tar_name}: {e}", file=sys.stderr)
            cleanup_extract(extract_dir)
            continue

        if bp_path is None:
            print(f"[rank {RANK}] WARNING: no .bp in {tar_name}", file=sys.stderr)
            cleanup_extract(extract_dir)
            continue

        t_extract  = time.perf_counter() - t_extract

        t_validate = time.perf_counter()
        result     = validate_bp_file(bp_path, args)
        t_validate = time.perf_counter() - t_validate

        cleanup_extract(extract_dir)

        if result is None:
            continue

        for key in rank:
            rank[key] += result[key]

        rank_bad_indices.append(result["bad_indices"])
        rank_non_synth_indices.append(result["non_synth_indices"])
        rank_min_dists.append(result["min_dist_all"])
        rank_energies_pa.append(result["energies_per_atom"])
        rank_synth_energies.append(result["synth_energies_pa"])

        if args.verbose:
            s1_pct = 100.0 * result["n_invalid"]       / max(result["n_total"], 1)
            s2_pct = 100.0 * result["n_synthesizable"]  / max(result["n_total"], 1)
            print(f"[rank {RANK:05d}] [{file_idx+1:4d}/{len(my_tars)}] {tar_name} "
                  f"total={result['n_total']} "
                  f"s1_invalid={result['n_invalid']}({s1_pct:.1f}%) "
                  f"synthesizable={result['n_synthesizable']}({s2_pct:.1f}%) "
                  f"extract={t_extract:.1f}s validate={t_validate:.1f}s",
                  flush=True)

    # Save per-rank .npz
    out_path = os.path.join(args.out_dir, f"validation_rank{RANK:05d}.npz")
    np.savez_compressed(
        out_path,
        **{k: np.array([v], dtype=np.int64) for k, v in rank.items()},
        bad_indices       = np.concatenate(rank_bad_indices)       if rank_bad_indices       else np.array([], dtype=np.int64),
        non_synth_indices = np.concatenate(rank_non_synth_indices) if rank_non_synth_indices else np.array([], dtype=np.int64),
        min_dist_all      = np.concatenate(rank_min_dists)         if rank_min_dists         else np.array([], dtype=np.float32),
        energies_per_atom = np.concatenate(rank_energies_pa)       if rank_energies_pa       else np.array([], dtype=np.float32),
        synth_energies_pa = np.concatenate(rank_synth_energies)    if rank_synth_energies    else np.array([], dtype=np.float32),
    )

    # Global reduction
    if COMM:
        COMM.Barrier()
        g = {k: COMM.reduce(v, op=MPI.SUM, root=0) for k, v in rank.items()}
    else:
        g = dict(rank)

    if RANK == 0:
        tot  = g["n_total"]
        s1ok = tot - g["n_invalid"]
        synt = g["n_synthesizable"]
        def pct(n): return 100.0 * n / max(tot, 1)

        print("\n" + "="*62)
        print("VALIDATION SUMMARY")
        print("="*62)
        print(f"  Total structures                : {tot:>15,}")
        print(f"")
        print(f"  STAGE 1 — Physical validity")
        print(f"  ├─ Passed                       : {s1ok:>15,}  ({pct(s1ok):.3f}%)")
        print(f"  ├─ Failed                       : {g['n_invalid']:>15,}  ({pct(g['n_invalid']):.3f}%)")
        print(f"  │    ├─ Energy out of range      : {g['fail_energy']:>15,}")
        print(f"  │    ├─ Force too large/NaN      : {g['fail_forces']:>15,}")
        print(f"  │    └─ Atom overlap (<{args.min_dist}A)    : {g['fail_mindist']:>15,}")
        print(f"")
        print(f"  STAGE 2 — Synthesizability")
        print(f"  ├─ Checked                      : {g['n_synth_checked']:>15,}")
        print(f"  ├─ Synthesizable                : {synt:>15,}  ({pct(synt):.3f}%)")
        print(f"  ├─ Failed                       : {g['n_synth_fail']:>15,}")
        print(f"  │    ├─ Non-synthesizable element: {g['fail_element']:>15,}")
        print(f"  │    ├─ Isolated atom            : {g['fail_isolated']:>15,}")
        print(f"  │    ├─ Valence exceeded         : {g['fail_valence']:>15,}")
        print(f"  │    └─ Radical detected         : {g['fail_radical']:>15,}")
        print(f"")
        print(f"  Per-rank .npz files in          : {args.out_dir}")
        print("="*62)


if __name__ == "__main__":
    main()
