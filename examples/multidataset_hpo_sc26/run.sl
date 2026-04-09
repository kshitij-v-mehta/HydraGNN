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

HYDRAGNN_ROOT=/lustre/orion/lrn070/world-shared/kmehta/hydragnn/hydragnn-fork
EXAMPLE_DIR=$HYDRAGNN_ROOT/examples/multidataset_hpo_sc26

export PYTHONPATH=$PYTHONPATH:$HYDRAGNN_ROOT

# Add ADIOS to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/lustre/orion/lrn070/world-shared/kmehta/hydragnn/hydragnn-libenv-installation/envs/hydragnn/lustre/orion/lrn070/world-shared/kmehta/hydragnn/hydragnn-libenv-installation/envs/hydragnn/lib/python3.11/site-packages/:

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

function cmd() {
    echo "$@"
    time "$@"
}


echo "===== Module List ====="
module list

echo "===== Check ====="
which python
python -c "import adios2; print(adios2.__version__, adios2.__file__)"
python -c "import torch; print(torch.__version__, torch.__file__)"

echo "===== LD_LIBRARY_PATH ====="
echo $LD_LIBRARY_PATH | tr ':' '\n'

export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p "$MIOPEN_USER_DB_PATH"

export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=1
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1

# Multi-node torch/c10d networking
MASTER_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_IP=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1 {print $1}')
if [ -z "$MASTER_IP" ]; then
    MASTER_IP="$MASTER_HOST"
fi
export MASTER_ADDR="$MASTER_IP"
export MASTER_PORT=${MASTER_PORT:-29501}
export HYDRAGNN_MASTER_ADDR="$MASTER_ADDR"
export HYDRAGNN_MASTER_PORT="$MASTER_PORT"
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_IFNAME=hsn0


# Keep key size/precision knobs aligned with job-sc26-oom.sh style
export BATCH_SIZE=50
export NUM_SAMPLES=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH*NUM_EPOCH))
export INFER_PRECISION=fp64

# Hard-coded training log directory to load for inference.
CHECKPOINT_LOGDIR="./logs/multidataset_hpo-BEST6-fp64"

if [ -z "$CHECKPOINT_LOGDIR" ] || [ ! -d "$CHECKPOINT_LOGDIR" ]; then
    echo "ERROR: Could not resolve CHECKPOINT_LOGDIR."
    echo "Expected hard-coded path: $CHECKPOINT_LOGDIR"
    exit 1
fi

if [ ! -f "$CHECKPOINT_LOGDIR/config.json" ]; then
    echo "ERROR: config.json not found in $CHECKPOINT_LOGDIR"
    exit 1
fi

echo "Using checkpoint log dir: $CHECKPOINT_LOGDIR"
echo "Precision:                $INFER_PRECISION"

srun -N $SLURM_JOB_NUM_NODES -n $SLURM_JOB_NUM_NODES bash -c "mkdir -p /tmp/kmehta/" 
srun -N $SLURM_JOB_NUM_NODES -n $SLURM_JOB_NUM_NODES bash -c "rm -rf  /tmp/kmehta/*" 

cmd srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest -l --kill-on-bad-exit=1 \
    --export=ALL,MASTER_ADDR=$MASTER_ADDR,MASTER_PORT=$MASTER_PORT,HYDRAGNN_MASTER_ADDR=$HYDRAGNN_MASTER_ADDR,HYDRAGNN_MASTER_PORT=$HYDRAGNN_MASTER_PORT,GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME,NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
    python -u "./inference_fused_write_adios.py" \
    --logdir "$CHECKPOINT_LOGDIR" \
    --num_structures 15050 \
    --min_atoms 20 \
    --max_atoms 500 \
    --batch_size $BATCH_SIZE \
    --precision $INFER_PRECISION
