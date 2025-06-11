#!/bin/bash
#SBATCH -p hgx
#SBATCH --gres=gpu:1 # Request 1 GPU
#SBATCH --job-name=retina_training
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --ntasks=1
#SBATCH --array=0-4 # Change based on number of configurations

n_runs=5

# Generate configurations dynamically
configurations=()
# Store experiment names
experiment_names=()

for run in $(seq 1 $n_runs); do
    experiment_name="EXP_REG_CONF_RUN_${run}"
    experiment_names+=("$experiment_name")
done

# Select the correct configuration based on SLURM_ARRAY_TASK_ID
EXP_NAME=${experiment_names[$SLURM_ARRAY_TASK_ID]}

# Create folder for slurm logs (specified at the top)
LOG_DIR="slurm_logs"
mkdir -p "$LOG_DIR"

# Activate the conda environment
export PATH="/home/inf148247/anaconda3/bin:$PATH"
source activate neural_env

wandb online

# Execute Python training script with selected configuration
python ./src/train.py \
        hydra.run.dir="results/${EXP_NAME}/"

python ./src/test.py \
        -cp "../results/${EXP_NAME}/" \
        hydra.run.dir=. \
        hydra.output_subdir=null \
        hydra/job_logging=disabled \
        hydra/hydra_logging=disabled \
        training.predictor.activation="sigmoid"

