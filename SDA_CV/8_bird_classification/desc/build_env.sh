#!/bin/bash -x

ENV_NAME=cv

conda create -y -n $ENV_NAME python=3.10.12

conda_install="conda install -n $ENV_NAME -y"

$conda_install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
$conda_install lightning matplotlib albumentations timm -c conda-forge
$conda_install ipykernel --update-deps --force-reinstall
$conda_install -c anaconda pandas pytest
