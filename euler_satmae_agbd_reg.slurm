#!/bin/bash
#SBATCH --account=es_schin
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=30:00:00
#SBATCH --job-name=satmae_agbd_reg_batch_64_and_32
#SBATCH --mail-type=BEGIN,END
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8

source /cluster/scratch/anogueira/miniconda3/bin/activate
conda init --all
conda activate pangaea-bench

# Debugging: print python and hydra info
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "hydra-core version: $(pip show hydra-core | grep Version)"
echo "PYTHONPATH: $PYTHONPATH"

cd /cluster/scratch/anogueira/pangaea-bench/

export HYDRA_FULL_ERROR=1

# Print GPU info for debugging
echo "GPU info:" && nvidia-smi

torchrun --nnodes=1 --nproc_per_node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 pangaea/run.py \
    --config-name=train \
    dataset=agbd \
    encoder=satmae_base \
    decoder=reg_upernet \
    preprocessing=reg_resize \
    criterion=mse \
    task=regression \
    use_wandb=True \
    batch_size=64 \
    test_batch_size=32