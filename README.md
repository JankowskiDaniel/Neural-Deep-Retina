# Neural Deep Retina #

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-360/)

## Environment setup ##

1. Create python 3.11 (or newer) environment: <br>
   `python3.11 -m venv venv`
2. Activate the environment: <br>
Linux: `source venv/bin/activate` <br>
Windows: `venv\Scripts\activate.bat`
3. Install required dependencies: <br>
`pip install -r requirements.txt`

Alternatively to point 3., install dependencies by listing only the core packages:

```sh
# Install torch with CUDA
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
# OR install torch with CPU only
pip install torch==2.3.1 torchvision==0.18.1
# Install other necessary packages
pip install tqdm==4.66.4 pyyaml==6.0.1 h5py==3.11.0 scikit-learn==1.5.1 seaborn==0.13.2
```


## Training ##

For running model training use the `src/train.py` script. It is assumed that the script is run at the root level of the repository (follow this convention to avoid directory conflicts).

The script accepts two parameters:

- `--results_dir` - a directory inside the `results` dir, where the trained models, config, and optionally logs are stored.
- `--config` (optional) - a config file name. Default to `config.yaml`

An exemplary command for running the training:

`python src/train.py --results_dir my_training` - the training results will be saved inside `results/my_training` directory using the `config.yaml` file.


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

  

