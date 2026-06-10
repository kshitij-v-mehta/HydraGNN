#!/usr/bin/env python3
"""
Count structures in the OMol25 neutral_train ASE dataset whose composition
is a subset of {H, C, N, O}.

MPI-sharded: each rank processes a contiguous slice of the global index range
and counts locally; counts are reduced to rank 0.

Usage (inside an Andes batch allocation):
    srun -n <ntasks> python count_chno.py --src /path/to/neutral_train
"""
import argparse
import sys
import time
import numpy as np
from mpi4py import MPI
from tqdm import tqdm
from fairchem.core.datasets import AseDBDataset

ALLOWED = frozenset((1, 6, 7, 8))  # H, C, N, O


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to neutral_train directory")
    ap.add_argument("--log-every", type=int, default=50000,
                    help="how often (in structures) to update the running "
                         "CHNO tally shown in the rank-0 progress bar")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    # Each rank opens the dataset independently. AseDBDataset over a directory
    # concatenates all shards and uses metadata.npz for cross-shard indexing.
    dataset = AseDBDataset({"src": args.src})
    n = len(dataset)

    # Contiguous block decomposition of [0, n) across ranks.
    lo = rank * n // nproc
    hi = (rank + 1) * n // nproc

    if rank == 0:
        print(f"[count_chno] total structures = {n}", flush=True)
        print(f"[count_chno] ranks = {nproc}, "
              f"~{(n + nproc - 1) // nproc} structures/rank", flush=True)

    t0 = time.time()
    local_count = 0
    local_seen = 0

    # Only rank 0 shows a progress bar (representative of overall pace).
    # Other ranks iterate without a bar to keep stderr clean.
    if rank == 0:
        iterator = tqdm(
            range(lo, hi),
            total=hi - lo,
            desc="rank0 CHNO scan",
            unit="struct",
            mininterval=2.0,        # don't spam the SLURM err file
            file=sys.stderr,
        )
    else:
        iterator = range(lo, hi)

    for i in iterator:
        atoms = dataset.get_atoms(i)
        z = set(int(x) for x in atoms.get_atomic_numbers())
        if z <= ALLOWED:           # subset test
            local_count += 1
        local_seen += 1

        # Surface the running CHNO tally in the bar's postfix.
        if rank == 0 and local_seen % args.log_every == 0:
            iterator.set_postfix(chno=local_count, refresh=False)

    # Reduce counts and the per-rank seen totals (sanity check coverage == n).
    total_count = comm.reduce(local_count, op=MPI.SUM, root=0)
    total_seen = comm.reduce(local_seen, op=MPI.SUM, root=0)

    # Gather per-rank timing for load-balance visibility.
    elapsed = time.time() - t0
    all_elapsed = comm.gather(elapsed, root=0)

    if rank == 0:
        print("=" * 60, flush=True)
        print(f"CHNO-only structures : {total_count}", flush=True)
        print(f"total scanned        : {total_seen}  (expected {n})", flush=True)
        frac = total_count / n * 100 if n else 0.0
        print(f"fraction CHNO-only   : {frac:.2f}%", flush=True)
        print(f"slowest rank         : {max(all_elapsed)/60:.1f} min", flush=True)
        print(f"fastest rank         : {min(all_elapsed)/60:.1f} min", flush=True)
        assert total_seen == n, "coverage mismatch — decomposition bug"


if __name__ == "__main__":
    main()

