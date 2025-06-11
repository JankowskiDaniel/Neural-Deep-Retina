#!/bin/bash
#SBATCH -p hgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-4 # Change based on number of configurations


n_runs=5

experiment_names=()

for run in $(seq 1 $n_runs); do
    experiment_name="EXP_REG_BEST_HPO_CFC_RUN_${run}"
    experiment_names+=("$experiment_name")
done

EXP_NAME=${experiment_names[$SLURM_ARRAY_TASK_ID]}

export PATH="/home/inf148247/anaconda3/bin:$PATH"
source activate neural_env

wandb online

# Execute Python training script with selected configuration
python ./src/train.py \
        hydra.run.dir="results/${EXP_NAME}/" \
        training.encoder.learning_rate=0.001 \
        training.predictor.name="SimpleCFC" \
        training.predictor.activation="relu" \
        training.predictor.learning_rate=0.0003735 \
        data.subseq_len=40 \
        training.loss_function="mse" \
        data.path="data/neural_code_data/retina/9_units.h5" \
        data.num_units=9 \
        data.subset_size=-1 \
        training.batch_size=2048 \
        testing.batch_size=2048 

python ./src/test.py \
        -cp "../results/${EXP_NAME}/" \
        hydra.run.dir=. \
        hydra.output_subdir=null \
        hydra/job_logging=disabled \
        hydra/hydra_logging=disabled \
        training.predictor.activation="relu"