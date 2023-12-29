#!/bin/bash -x

# ---> Install it first to solve env faster <---
# conda install -n base conda-libmamba-solver
# conda config --set solver libmamba

ENV_NAME=cv

conda create -y -n $ENV_NAME python=3.10.12

conda_install="conda install -n $ENV_NAME -y"

$conda_install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
$conda_install lightning matplotlib albumentations timm tensorboard -c conda-forge
$conda_install ipykernel --update-deps --force-reinstall
$conda_install -c anaconda pandas pytest
