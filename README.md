# Multimeasurement Generative Models

This repository contains the code for [Multimeasurement Generative Models](https://openreview.net/forum?id=QRX0nCX_gk) published at ICLR 2022, implemented in PyTorch 1.9.0. 

## Setup

System pre-requisites: 
- Anaconda or [Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html)
- CUDA drivers that support CUDA 11.1 (optional, but probably necessary for experiments other than MNIST)

```bash
# Clone this repository
git clone https://github.com/nnaisense/mems.git
cd mems
# Create and activate a conda environment
conda env create -f env.yml && conda activate mems
# Install the project in editable mode.
pip install -e .
```

Notes: 
- The required dataset will be downloaded when the first experiment starts, into the location configured in the config file (`data.data_dir`), unless it already exists.
- To log training to neptune, `pip install neptune-client`, set the API token and then set the `trainer.neptune` config to a neptune project. Otherwise, CSVLogger will be used as fallback.
- The code will generate images using MCMC during training and save them using the logger, according to the config in `model.training.sampling`. To disable this, you can set `model.training.sampling.mode=None`.

## Training

```bash
# Train a 1x16 MDAE on MNIST 
CUDA_VISIBLE_DEVICES=0 python run/train.py with configs/mdae_mnist_1x16.yaml -f
# Train a 1x8 MDAE on CIFAR10 
CUDA_VISIBLE_DEVICES=0,1,2,3 python run/train.py with configs/mdae_cifar10_1x8.yaml -f
# Train a 4x8 MDAE on FFHQ256
CUDA_VISIBLE_DEVICES=0,1,2,3 python run/train.py with configs/mdae_ffhq256_4x8.yaml -f
# Train a 1x4 MUVB on MNIST
CUDA_VISIBLE_DEVICES=0 python run/train.py with configs/muvb_mnist_1x4.yaml -f

# You can modify any config entry from the command line, e.g.,
# Train a 0.5x4 MDAE on MNIST 
python run/train.py with configs/mdae_mnist_1x16.yaml model.sigma=[0.5] model.M=[4] -f
```

## Authors / Citation
- [**Saeed Saremi**](https://github.com/saeedsaremi)
- [**Rupesh Srivastava**](https://rupeshks.cc/)

```
Saremi, S., & Srivastava, R. K. (2022). Multimeasurement Generative Models. International Conference on Learning Representations. https://openreview.net/forum?id=QRX0nCX_gk 
```

```
@inproceedings{saremi2022multimeasurement,
	title        = {Multimeasurement Generative Models},
	author       = {Saeed Saremi and Rupesh Kumar Srivastava},
	year         = 2022,
	booktitle    = {International Conference on Learning Representations},
	url          = {https://openreview.net/forum?id=QRX0nCX_gk}
}
```
