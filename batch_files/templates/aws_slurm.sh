#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --account=laion
#SBATCH --nodes={node_count}
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --time=200:00:00
#SBATCH --chdir={launch_dir}
#SBATCH --output=/path/to/out
#SBATCH --error=/path/to/err
#SBATCH --requeue
#SBATCH --exclusive


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
    /admin/home-vkramanuj/miniconda3/envs/ldm/bin/python -u train.py {option dotlist}
    


