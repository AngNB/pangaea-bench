#!/bin/bash
#SBATCH --account=es_schin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=00:20:00
#SBATCH --job-name=satmae_hlsburnscars_seg
#SBATCH --mail-type=BEGIN,END
#SBATCH --mem-per-cpu=4096

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
    dataset=hlsburnscars \
    encoder=remoteclip \
    decoder=seg_upernet\
    preprocessing=seg_default \
    criterion=cross_entropy \
    task=segmentation \
    use_wandb=True