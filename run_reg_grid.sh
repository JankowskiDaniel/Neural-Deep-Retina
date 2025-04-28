#!/bin/bash
#SBATCH -p hgx
#SBATCH --gres=gpu:1 # Request 1 GPU
#SBATCH --job-name=retina_training
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --mem-per-gpu=16G
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-179 # Change based on number of configurations

# Define grid search parameters
predictors=("SimpleCFC" "SimpleLTC", "SingleLSTM")
lrs=(0.002 0.002 0.0005)
subseq_lengths=(40 20)
loss_functions=("mse" "mae")
datasets=("data/neural_code_data/retina/9_units.h5"
          "data/neural_code_data/retina/14_units.h5"
          "data/neural_code_data/retina/27_units.h5")
num_units=(9 14 27)
n_runs=5

# Generate configurations dynamically
configurations=()
# Store experiment names
experiment_names=()
for p in $(seq 0 $((${#predictors[@]} - 1))); do
  for subseq_len in "${subseq_lengths[@]}"; do
    for loss in "${loss_functions[@]}"; do
      for i in $(seq 0 $((${#datasets[@]} - 1))); do
          for run in $(seq 1 $n_runs); do
            dataset=${datasets[$i]}
            # Dataset will be referenced by num_units
            num_unit=${num_units[$i]}
            predictor=${predictors[$p]}
            lr=${lrs[$p]}
            # Create a unique name for each run
            experiment_name="EXP_REG_${predictor}_${subseq_len}_${loss}_${num_unit}_RUN_${run}"
            experiment_names+=("$experiment_name")
            # Add the configuration to the array
            configurations+=("$predictor $lr $subseq_len $loss $dataset $num_unit")
          done
        done
      done
  done
done

# Select the correct configuration based on SLURM_ARRAY_TASK_ID
CONFIG=(${configurations[$SLURM_ARRAY_TASK_ID]})
EXP_NAME=${experiment_names[$SLURM_ARRAY_TASK_ID]}

# Create folder for slurm logs (specified at the top)
LOG_DIR="slurm_logs"
mkdir -p "$LOG_DIR"

echo "Running configuration: ${CONFIG[@]}"

# Activate the conda environment
export PATH="/home/inf148247/anaconda3/bin:$PATH"
source activate neural_env

wandb online

# Execute Python training script with selected configuration
python ./src/train.py \
        hydra.run.dir="results/${EXP_NAME}/" \
        training.predictor.name="${CONFIG[0]}" \
        training.predictor.learning_rate="${CONFIG[1]}" \
        data.subseq_len="${CONFIG[2]}" \
        training.loss_function="${CONFIG[3]}" \
        data.path="${CONFIG[4]}" \
        data.num_units="${CONFIG[5]}" \
        data.subset_size=-1 \
        training.batch_size=4096 \
        testing.batch_size=4096 

python ./src/test.py \
        -cp "../results/${EXP_NAME}/" \
        hydra.run.dir=. \
        hydra.output_subdir=null \
        hydra/job_logging=disabled \
        hydra/hydra_logging=disabled


