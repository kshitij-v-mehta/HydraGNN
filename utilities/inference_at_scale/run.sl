#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J inference_file_io
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 0:59:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END


# Conda activate
eval "$(conda shell.bash hook)"    # or: source /path/to/miniconda3/etc/profile.d/conda.sh
source /lustre/orion/lrn070/world-shared/kmehta/hydragnn//hydragnn-libenv-installation/envs/hydragnn/bin/activate

# Add ADIOS to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/lustre/orion/lrn070/world-shared/kmehta/hydragnn/hydragnn-libenv-installation/envs/hydragnn/lustre/orion/lrn070/world-shared/kmehta/hydragnn/hydragnn-libenv-installation/envs/hydragnn/lib/python3.11/site-packages/:


set -x

# Create PyG structures and write them to /mnt/bb
time srun -n $((SLURM_JOB_NUM_NODES*8)) -c7 python3 ./write_structures.py

# Combine the 8 files in /tmp into a new file in /mnt/bb/
time srun -n $SLURM_JOB_NUM_NODES -N $SLURM_JOB_NUM_NODES python3 -u ./combine_adios.py

# Copy the combined file to Orion
time srun -n $SLURM_JOB_NUM_NODES -N $SLURM_JOB_NUM_NODES bash -c "cp -r /mnt/bb/kmehta/structures-all-*.bp ."

set +x

echo -e "ALL DONE. GOODBYE"

