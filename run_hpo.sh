#!/bin/bash
#SBATCH -p hgx
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
export PATH="/home/inf148247/anaconda3/bin:$PATH"
source activate neural_env

wandb online

# Here goes the generated command to run the agent
wandb agent jankowskidaniel06-put/Neural-Deep-Retina-src/e9f7vc0s
