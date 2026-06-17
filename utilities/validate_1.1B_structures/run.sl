#!/bin/bash
#SBATCH --job-name=validate_structures
#SBATCH --account=
#SBATCH --partition=batch
#SBATCH --nodes=40
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=validate_%j.out
#SBATCH --error=validate_%j.err

# ============================================================
# Configuration — edit these paths
# ============================================================
TAR_DIR="/PATH/TO/TAR_FILES"
OUT_DIR="/PATH/TO/validation_results"
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# ============================================================
# Environment
# ============================================================
module load
source activate

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "Job:          ${SLURM_JOB_ID}"
echo "Nodes:        ${SLURM_NNODES}"
echo "Tasks:        ${SLURM_NTASKS}  (${SLURM_NTASKS_PER_NODE}/node)"
echo "Tar directory: ${TAR_DIR}"
echo "Output dir:   ${OUT_DIR}"
echo "Start:        $(date)"
echo "============================================================"

# /tmp is node-local on Andes — each rank writes to a unique subdir
# so 32 ranks/node never collide even though they share /tmp
srun python "${SCRIPT_DIR}/validate_structures.py" \
    --tar_dir   "${TAR_DIR}" \
    --out_dir   "${OUT_DIR}" \
    --tmp_dir   /tmp \
    --min_dist  0.5 \
    --max_force 50.0 \
    --energy_lo -20.0 \
    --energy_hi  20.0 \
    --verbose

echo "End: $(date)"
