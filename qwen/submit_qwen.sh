#!/bin/bash
#SBATCH --job-name        polargrad_qwen
#SBATCH --nodes           1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task   16
#SBATCH --mem             64gb
#SBATCH --gpus            h100:1
#SBATCH --time            6:00:00
#SBATCH --output          /scratch/marque6/logs/%x_%j.out
#SBATCH --error           /scratch/marque6/logs/%x_%j.err

# -----------------------------------------------------------------------
OPTIMIZER=${1:-adamw}
scontrol update JobId=$SLURM_JOB_ID JobName="polargrad_qwen_${OPTIMIZER}"

export HF_HOME=/scratch/marque6/hf_cache
mkdir -p $HF_HOME /scratch/marque6/logs

export PATH="$HOME/.pixi/bin:$PATH"

cd $SLURM_SUBMIT_DIR

PYTHON=$(pixi run which python)
echo "Using Python: $PYTHON"

echo "Starting train_qwen.py"
srun $PYTHON train_qwen.py --optimizer ${1:-adamw} #adamw, muon_polarsgdm, muon_adamw 

