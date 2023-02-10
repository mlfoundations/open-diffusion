
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


export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802

srun --cpu_bind=v --accel-bind=gn \
    /admin/home-vkramanuj/miniconda3/envs/ldm/bin/python -u train.py {option dotlist}