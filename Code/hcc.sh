#!/bin/bash
#SBATCH --job-name=sens_run
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=deep
#SBATCH --partition=deep
#SBATCH --gres=gpu:1

# Set up conda
source /sailhome/malvlai/anaconda3/etc/profile.d/conda.sh
conda activate CostEffective

# Ensure logs dir exists
mkdir -p logs

# Run the script
python hccDistributionAnalysis.py
