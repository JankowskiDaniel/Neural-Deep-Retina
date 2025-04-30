# Neural Deep Retina #

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-360/)

## Environment setup ##

1. Create python 3.11 (or newer) environment: <br>
   `python3.11 -m venv venv`
1. Activate the environment: <br>
Linux: `source venv/bin/activate` <br>
Windows: `venv\Scripts\activate.bat`
1. Install the project: `pip install -e .`.
1. Altenatively, install requirements manually: 
    ```sh
    # Install torch with CUDA
    pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
    # OR install CPU only torch 
    pip install torch==2.3.1 torchvision==0.18.1
    # Then install other dependencies
    pip install -r requirements.txt
    # Optionally, install jupyter notebook dependencies
    pip install ipykernel==6.29.5 ipython==8.26.0
    ```

Code tests can be executed using `pytest`.

## Dataset ##

The data set used in this project can be found [here](https://purl.stanford.edu/rk663dm5577). Please download and unzip in the `data` directory.

We manually create a validation split for the evaluation. This can be done using `manipulate_data.py`. For example, open the terminal in the root directory an run:

```bash
python -m data.manipulate_data data/neural_code_data/ganglion_cell_data/15-10-07/naturalscene.h5 --man_type make_val_split --train_ratio 0.8
```

This will create a new file named `naturalscene_with_val.h5`, where the first 80% of the original data will constitue the new train set, and the rest will become the validation set. The test set is copied to the new file without modifications.

You can also add Gaussian noise to the selected subset of stimulus (default is test):

```bash
python -m data.manipulate_data data/neural_code_data/ganglion_cell_data/15-10-07/naturalscene.h5 --man_type add_noise --sigma 0.3
```

## Training ##

For running model training, use the `src/train.py` script. It is assumed that the script is run at the root level of the repository (follow this convention to avoid directory conflicts).

Specify `results_dir` in the config file under `hydra.run.dir`. 

An exemplary command for running the training:

```sh
python src/train.py
``` 


The training results will be saved inside `results/[results_dir]` directory and training will proceed according to `config.yaml` file.

### Running training in the slurm cluster ###

1. Export the path to the conda
```sh
export PATH="/home/inf148247/anaconda3/bin:$PATH"
```
2. Activate the environment
```sh
source activate neural_env
```
3. Run the training in the background by submitting a job
```sh
sbatch run_training.sh
```
4. Or connect to the hgx cluster in an interactive mode:
```sh
srun -p hgx -w hgx1 --gres=gpu:1 --pty /bin/bash -l
```

### Using pre-trained models ###

For setting up pre-trained models, please check out the readme in the `pretrained_weights` folder.

## Testing ##

For testing, use the `src/test.py` script. Again, it is assumed that the script is run at the root level of the repository.

An exemplary command for running the testing (uses hydra):

```bash
python .\src\test.py -cp ../results/test_hydra/ hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled
```

#### *Important* ####

Run testing only after training is finished and the models are saved. By default, the model state dict is loaded from `results/results_dir/models/best.pth`.

Testing parameters are specified in the `config.yaml` file present in the `results_dir` directory. The config file is copied there during training.

  

