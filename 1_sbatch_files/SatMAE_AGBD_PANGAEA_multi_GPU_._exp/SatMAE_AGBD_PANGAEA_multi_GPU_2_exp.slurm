#!/bin/bash
#SBATCH --job-name=SatMAE_AGBD_PANGAEA_2_exp
#SBATCH --account=es_schin
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=300G
#SBATCH --gpus=rtx_4090:8
#SBATCH --mail-type=END,FAIL

# --- HEADER AND STRATEGY ---
# Print job info and explain the workflow
echo "################################################################################"
echo "# AGBD PANGAEA TRAINING with temp dir"
echo "# Job: ${SLURM_JOB_ID}"
echo "# Node: $(hostname), Date: $(date)"
echo "################################################################################"

set -euo pipefail

# Load ETH proxy for internet access
module load eth_proxy

user=$(whoami)

echo "Running on ETH Euler cluster as user: $user"

################################################################################
# STAGE 1: COMPREHENSIVE DATA STAGING
################################################################################

echo ""
echo "STAGE 1: Staging data to TMPDIR"

mkdir -p "$TMPDIR/agbd_data"
mkdir -p "$TMPDIR/agbd_splits" 
mkdir -p "$TMPDIR/workspace"
mkdir -p "$TMPDIR/logs"

echo "   TMPDIR structure created:"
echo "   Data: $TMPDIR/agbd_data"
echo "   Splits: $TMPDIR/agbd_splits"
echo "   Workspace: $TMPDIR/workspace"
echo "   Logs: $TMPDIR/logs"

# Stage AGBD data
echo "Staging AGBD HDF5 files..."
rsync --include '*v4_*-20.h5' --exclude '*' -av \
    /cluster/work/igp_psr/gsialelli/Data/patches/ \
    "$TMPDIR/agbd_data/" || {
    echo "Failed to stage HDF5 files"
    exit 1
}

echo "Staging corresponding csv files (for train) ..."
rsync -av \
    /cluster/work/igp_psr/gsialelli/Data/patches/train_features_2019.csv \
    "$TMPDIR/agbd_data/" || {
    echo "Failed to stage csv files (for train) files"
    exit 1
}

echo "Staging corresponding csv files (for val) ..."
rsync -av \
    /cluster/work/igp_psr/gsialelli/Data/patches/val_features_2019.csv \
    "$TMPDIR/agbd_data/" || {
    echo "Failed to stage csv files (for val) files"
    exit 1
}

echo "Staging corresponding csv files (for test) ..."
rsync -av \
    /cluster/work/igp_psr/gsialelli/Data/patches/test_features_2019.csv \
    "$TMPDIR/agbd_data/" || {
    echo "Failed to stage csv files (for test) files"
    exit 1
}

echo "Staging statistics file..."
rsync -av \
    /cluster/work/igp_psr/gsialelli/Data/patches/statistics_subset_2019-2020-v4_new.pkl \
    "$TMPDIR/agbd_data/" || {
    echo "Failed to stage statistics file"
    exit 1
}

echo "Staging mapping (splits) file..."
rsync -av \
    /cluster/work/igp_psr/gsialelli/Data/AGB/biomes_splits_to_name.pkl \
    "$TMPDIR/agbd_splits/" || {
    echo "Failed to stage splits file"
    exit 1
}

echo "Data staging completed"

################################################################################
# STAGE 2: WORKSPACE SETUP (INTERNET-ENABLED APPROACH)
################################################################################

echo ""
echo "STAGE 2: Setting up workspace with internet access..."

PANGAEA_HOME="/cluster/home/$user/pangaea-bench"
VENV_HOME="/cluster/home/$user/pangaea-bench-venv3"

# Verify pangaea-bench exists in home directory
if [ ! -d "$PANGAEA_HOME" ]; then
    echo "pangaea-bench not found in home directory: $PANGAEA_HOME"
    exit 1
fi

# Verify virtual environment exists                                    <-------------------------------YOU MIGHT WANT TO CHANGE THIS TO YOUR VENV PATH OR JUST DELETE AND SEE BELOW
if [ ! -d "$VENV_HOME" ]; then
    echo "Virtual environment not found: $VENV_HOME"
    exit 1
fi

echo "Using pangaea-bench from home directory (with internet access)"
echo "Pangaea path: $PANGAEA_HOME"
echo "Virtual env: $VENV_HOME"

################################################################################
# STAGE 3: ENVIRONMENT ACTIVATION AND CONFIGURATION
################################################################################

echo ""
echo "STAGE 3: Activating environment from home directory..."

# Change to pangaea-bench in home directory (with internet access)
cd "$PANGAEA_HOME"
echo "Working directory: $(pwd)"

# Activate virtual environment from home directory                                           <--------------- NOT SURE BUT FOR CONDA JUST DO CONDA ACTIVATE PANGAEA-BENCH (or whatever you called it)
source "$VENV_HOME/bin/activate"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
module load stack/2024-06
module load openblas/0.3.24

# Set environment variables for proper path resolution
export HYDRA_FULL_ERROR=1

# Detect GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1 2>/dev/null || echo "Unknown")
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 2>/dev/null || echo "0")

echo "Detected GPU: $GPU_NAME"
echo "GPU Memory: ${GPU_MEMORY}MB"

################################################################################
# STAGE 5: TRAINING EXECUTION
################################################################################

#                                                                                         <------------- Overwrite this with your training command important: always add a space before the backslash to avoid syntax errors

TRAIN_CMD="torchrun --nnodes=1 --nproc_per_node=8 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 pangaea/run.py \
    --config-name=train \
    dataset=agbd \
    encoder=satmae_base \
    decoder=reg_upernet \
    preprocessing=reg_agbd_resize \
    dataset.debug=False \
    criterion=mse \
    task=regression \
    use_wandb=true \
    batch_size=128 \
    test_batch_size=128 \
    num_workers=16 \
    test_num_workers=16 \
    task.trainer.n_epochs=10 \
    task.evaluator.inference_mode=whole \
    dataset.img_size=25 \
    task.trainer.ckpt_interval=1 \
    task.trainer.eval_interval=1 \
    dataset.root_path=$TMPDIR/agbd_data \
    dataset.hdf5_dir=$TMPDIR/agbd_data \
    dataset.mapping_path=$TMPDIR/agbd_splits \
    dataset.norm_path=$TMPDIR/agbd_data \
    seed=75 \
    optimizer.lr=0.0001"


echo "Training command "
echo "$TRAIN_CMD" | tr '\\' '\\n' | sed 's/^/    /'

# -------- LOGGING AND EXECUTION --------
PERSISTENT_OUTPUT_DIR="/cluster/scratch/$user/outputs/${SLURM_JOB_ID}/"
mkdir -p "$PERSISTENT_OUTPUT_DIR"

echo "$TRAIN_CMD" > "$PERSISTENT_OUTPUT_DIR/command.txt"

set +e  # Let the script continue on error to handle logging
eval "$TRAIN_CMD" > "$PERSISTENT_OUTPUT_DIR/training.log" 2>&1
EXIT_CODE=$?
set -e

################################################################################
# STAGE 6: RESULTS COLLECTION AND CLEANUP
################################################################################

if [ -d "outputs" ]; then
    echo "Copying outputs to persistent storage..."
    mkdir -p "$PERSISTENT_OUTPUT_DIR/outputs"
    rsync -av outputs/ "$PERSISTENT_OUTPUT_DIR/outputs/" || echo "Failed to copy outputs"
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training finished successfully!" | tee -a "$PERSISTENT_OUTPUT_DIR/training.log"
else
    echo "TRAINING FAILED with exit code $EXIT_CODE" | tee -a "$PERSISTENT_OUTPUT_DIR/training.log"
    exit $EXIT_CODE
fi

# Usage: just sbatch whateveryouname.slurm, then squeue --> gives batch id look for slurm-batch_id.out in home directory and tail -f slurm-batch_id.out or cat slurm-batch_id.out to see the output
# The download takes around 20 minutes.... but should run faster now fingers crossed. I also try to optimize the data staging and try yet another approach (loading the WHOLE dataset into RAM, if I can do this we should be 10x faster haha but absolutely no guarantees...)