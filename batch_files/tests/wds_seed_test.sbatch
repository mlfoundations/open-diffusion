#!/bin/bash
#SBATCH --job-name=wds-sample-test
#SBATCH --partition=g40423
#SBATCH --account=laion
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --time=96:00:00
#SBATCH --chdir=/admin/home-vkramanuj/sd-training
#SBATCH --output=/fsx/home-vkramanuj/resample-test.out
#SBATCH --error=/fsx/home-vkramanuj/resample-test.err
#SBATCH --requeue
#SBATCH --exclusive
#SBATCH --comment=laion 


module load cuda/11.7
export NCCL_PROTO=simple

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export NCCL_DEBUG=info
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0
export TORCH_SHOW_CPP_STACKTRACES=1


srun --comment laion --cpu_bind=v --accel-bind=gn \
    /admin/home-vkramanuj/miniconda3/envs/ldm/bin/python -u scripts/test_resampled_shards.py 
