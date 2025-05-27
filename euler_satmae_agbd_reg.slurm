#!/bin/bash
#SBATCH --account=es_schin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=30:00:00
#SBATCH --job-name=satmae_agbd_reg
#SBATCH --mail-type=BEGIN,END
#SBATCH --mem-per-cpu=4096

# module load stack/2024-06
# module load python/3.11.6
source /cluster/scratch/anogueira/pangaea-bench-venv/bin/activate

# Debugging: print python and hydra info
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "hydra-core version: $(pip show hydra-core | grep Version)"
echo "PYTHONPATH: $PYTHONPATH"

cd /cluster/scratch/anogueira/pangaea-bench/

export HYDRA_FULL_ERROR=1

# Print GPU info for debugging
echo "GPU info:" && nvidia-smi

/cluster/scratch/anogueira/pangaea-bench-venv/bin/torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
    --config-name=train \
    dataset=agbd \
    encoder=satmae_base \
    decoder=reg_upernet \
    preprocessing=reg_resize \
    criterion=mse \
    task=regression \
    use_wandb=True