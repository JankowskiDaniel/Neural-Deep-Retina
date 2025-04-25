#!/bin/bash
#SBATCH -w hgx2
#SBATCH -p hgx
#SBATCH --gres=gpu:1 # Request 1 GPU
#SBATCH --job-name=retina_training
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-3  # TODO Change based on number of configurations

# Define grid search parameters
predictors=("SimpleCfC" "SimpleLTC", "SimpleLSTM")
subseq_lengths=(40, 20)
loss_functions=("mse" "mae", "bce") # TODO
datasets=("data\neural_code_data\ganglion_cell_data\15-10-07\naturalscene_with_val.h5") # TODO
num_units=(9, 14, 26)
n_runs=5

# Generate configurations dynamically
configurations=()
# Store experiment names
experiment_names=()
for predictor in "${predictors[@]}"; do
  for subseq_len in "${subseq_lengths[@]}"; do
    for loss in "${loss_functions[@]}"; do
      for i in $(seq 0 $((${#datasets[@]} - 1))); do
          for run in $(seq 1 $n_runs); do
            dataset=${datasets[$i]}
            # Dataset will be referenced by num_units
            num_unit=${num_units[$i]}
            # Create a unique name for each run
            experiment_name="EXP_REG_${predictor}_${subseq_len}_${loss}_${num_unit}_RUN_${run}"
            experiment_names+=("$experiment_name")
            # Add the configuration to the array
            configurations+=("$predictor $subseq_len $loss $dataset $num_unit")
        done
      done
  done
done

# Select the correct configuration based on SLURM_ARRAY_TASK_ID
CONFIG=(${configurations[$SLURM_ARRAY_TASK_ID]})
EXP_NAME=${experiment_names[$SLURM_ARRAY_TASK_ID]}

echo "Running configuration: ${CONFIG[@]}"

# Activate the conda environment
export PATH="/home/inf148247/anaconda3/bin:$PATH"
source activate neural_env

# Execute Python training script with selected configuration
python .\src\train.py \
        hydra.run.dir="results/${EXP_NAME}/" \
        training.predictor="${CONFIG[0]}" \
        training.subseq_len="${CONFIG[1]}" \
        training.loss_function="${CONFIG[2]}" \
        data.path="${CONFIG[3]}" \
        data.num_units="${CONFIG[4]}" \

python .\src\test.py \
        -cp "../results/${EXP_NAME}/" \
        hydra.run.dir=. \
        hydra.output_subdir=null \
        hydra/job_logging=disabled \
        hydra/hydra_logging=disabled \


