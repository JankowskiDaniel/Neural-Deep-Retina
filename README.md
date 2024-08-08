# Neural Deep Retina #

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-360/)

## Environment setup ##

1. Create python 3.11 (or newer) environment: <br>
   `python3.11 -m venv venv`
2. Activate the environment: <br>
Linux: `source venv/bin/activate` <br>
Windows: `venv\Scripts\activate.bat`
3. Install required dependencies: 
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

## Training ##

For running model training, use the `src/train.py` script. It is assumed that the script is run at the root level of the repository (follow this convention to avoid directory conflicts).

The script accepts two parameters:

- `--results_dir` - a directory inside the `results` folder, where the saved models, config, and optionally logs are stored.
- `--config` (optional) - a config file name. Defaults to `config.yaml`

An exemplary command for running the training:

```sh
python src/train.py --results_dir my_training
``` 
The training results will be saved inside `results/my_training` directory and training will proceed according to `config.yaml` file.

## Testing ##

For testing, use the `src/test.py` script. Again, it is assumed that the script is run at the root level of the repository.

The script accepts one parameter:

- `--results_dir` - the same directory you provided for training

#### *Important* ####

Run testing only after training is finished and the models are saved. By default, the model state dict is loaded from `results/results_dir/models/best.pth`.

Testing parameters are specified in the `config.yaml` file present in the `results_dir` directory. The config file is copied there during training.

An exemplary command for running the testing:

```sh
python src/test.py --results_dir my_training
```

## Dataset ##

The default expected folder structure inside the `data` directory:
```
neural_code_data/
  ganglion_cell_data/
    15-10-07/
      naturalscene.h5
      whitenoise.h5
    15-11-21a/
      ...
    15-11-21b/
      ...
  interneuron_data/
    ...
```

  

