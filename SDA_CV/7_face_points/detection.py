from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
import albumentations as A

import numpy as np
import cv2

import os
from pathlib import Path
from PIL import Image

from dataclasses import dataclass
import matplotlib.pyplot as plt

SEPARATE_TRAIN_TEST=False

def OpenImg(path, dtype) -> np.ndarray:
    with Image.open(path) as img_file:
        img = img_file.convert("RGB")
        return np.array(img, dtype=dtype)

class FacePointsDataset(Dataset):
    def __init__(
        self,
        mode: str,
        model_input_shape,  # Example: (100, 100)
        img_dir: Path,
        train_gt: dict = None,
        fraction: float = 0.8,
        transform=None,
        is_full_load_on_start=False,
        fraction_fast_train: float = None
    ):
        assert mode in ["train", "val", "inf"]

        self.is_inference = mode == "inf"
        self.model_input_shape = model_input_shape
        self.img_dir = Path(img_dir)
        self._transform = transform

        img_names = sorted(os.listdir(img_dir))
        if SEPARATE_TRAIN_TEST:
            match mode:
                case "train":
                    self.img_names = img_names[: int(fraction * len(img_names))]
                case "val":
                    self.img_names = img_names[int(fraction * len(img_names)): ]
        else:
            if fraction_fast_train is None:
                self.img_names = img_names
            else:
                self.img_names = img_names[: int(fraction_fast_train * len(img_names))]

        self.is_full_load_on_start = is_full_load_on_start

        if self.is_inference:
            return
        
        self.points = [ train_gt[name].astype(np.float32)
                        for name in self.img_names ]
        
        if is_full_load_on_start:
            num_imgs = len(self.img_names)
            h, w = model_input_shape
            self.imgs = np.zeros((num_imgs, h, w, 3), dtype=np.float32)

            for i in range(num_imgs):
                img_path = self.img_dir / self.img_names[i]
                image = OpenImg(img_path, np.float32)

                if not self.is_inference:
                    self.points[i][0::2] *= h / image.shape[0]
                    self.points[i][1::2] *= w / image.shape[1]

                # Strange resize, todo: optimize
                image = torch.from_numpy(image).permute(2, 0, 1)
                image = torchvision.transforms.functional.resize(
                    image, self.model_input_shape, antialias=True,
                )
                self.imgs[i] = image.permute(1, 2, 0).numpy()


    def __len__(self):
        return len(self.img_names)

    def __getitem_train_val(self, index):
        if self.is_full_load_on_start:
            image = self.imgs[index]
        else:
            image = OpenImg(self.img_dir / self.img_names[index], np.float32)

        points = self.points[index].copy()

        if self._transform is not None:
            image = image.astype(np.uint8)
            res = self._transform(image=image, keypoints=points.reshape(-1, 2))
            tr_image, tr_points = res["image"], res["keypoints"]

            tr_points = np.array(tr_points, dtype=np.float32).flatten()
            if len(tr_points) == len(points):
                image, points = tr_image, tr_points

            image = image.astype(np.float32)

        assert image.dtype == np.float32
        assert points.dtype == np.float32

        image_orig_size = image.shape
        h, w = self.model_input_shape
        
        
        image = torch.from_numpy(image).permute(2, 0, 1)

        if not self.is_full_load_on_start:
            image = torchvision.transforms.functional.resize(
                image, self.model_input_shape, antialias=True
            )

            points[0::2] *= h / image_orig_size[0]
            points[1::2] *= w / image_orig_size[1]

        return image, points
    
    
    def __getitem_inference(self, index):
        if self.is_full_load_on_start:
            raise "not impl"
        
        image = OpenImg(self.img_dir / self.img_names[index], np.float32)

        assert image.dtype == np.float32

        image_orig_size = image.shape
        h, w = self.model_input_shape
        inv_scale_coefs = (image_orig_size[0] / h, image_orig_size[1] / w)
        
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = torchvision.transforms.functional.resize(
            image, self.model_input_shape, antialias=True
        )
        
        return image, inv_scale_coefs, self.img_names[index]
    
    def __getitem__(self, index):
        if self.is_inference:
            return self.__getitem_inference(index)
        else:
            return self.__getitem_train_val(index)

class ConvAct(nn.Sequential):
    def __init__(self, out_channels:int, kernel_size: int,
                 in_channels:int=None, stride=1):
        super().__init__()

        padding = (kernel_size - 1) // 2
        if in_channels is None:
            self.conv = nn.LazyConv2d(out_channels, kernel_size,
                                      padding=padding,stride=stride)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  padding=padding,stride=stride)
        self.bn = nn.LazyBatchNorm2d()
        self.act = nn.ReLU()

class Inception(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1x1 = ConvAct(64, 1)
        self.conv3x3 = nn.Sequential(
            ConvAct(64, 1),
            ConvAct(96, 3),
            ConvAct(96, 3)
        )
        self.conv5x5 = nn.Sequential(
            ConvAct(64, 1),
            ConvAct(96, 5)
        )
        self.pool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvAct(32, 1)
        )
        self.m = ConvAct(32, 3)

    def forward(self, inputs):
        batch_1x1 = self.conv1x1(inputs)
        batch_3x3 = self.conv3x3(inputs)
        batch_5x5 = self.conv5x5(inputs)
        batch_pool = self.pool(inputs)
        return torch.cat([batch_1x1, batch_3x3, batch_5x5, batch_pool], dim=1)

class PointDetector(nn.Module):
    def __init__(self, input_shape, num_points: int = 14 * 2):
        super().__init__()

        self.input_shape = input_shape
        self.model = nn.Sequential(
            ConvAct(32, 3, in_channels=3, stride=2),
            ConvAct(32, 3),
            ConvAct(64, 3),
            ConvAct(64, 3),
            nn.MaxPool2d(2, 2),

            ConvAct(80, 1),
            ConvAct(128, 3),
            # ConvAct(150, 3),
            ConvAct(192, 3),
            nn.MaxPool2d(2, 2),

            Inception(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),

            nn.LazyLinear(512),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),

            nn.LazyLinear(512),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),

            nn.LazyLinear(num_points)
        )

    def forward(self, X):
        input_shape = X.shape[2:] # [BatchSize, Colors, H, W] -> [H, W]
        assert self.input_shape ==  input_shape
        return self.model(X)

class TrainingPointDetector(pl.LightningModule):
    def __init__(self, input_shape, num_points=14 * 2, lr=1e-3):
        super().__init__()

        self.input_shape = input_shape
        self.lr = lr
        self.model = PointDetector(input_shape)
        self.loss = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        images, gt_coords = batch

        pred_coords = self.model(images)
        loss = self.loss(pred_coords, gt_coords)

        metrics = {"train_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True,
                      logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
          self.parameters(),
          lr=self.lr,
          weight_decay=1e-6
        )

        is_scheduler_enabled = True

        if is_scheduler_enabled:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=0.3,
                patience=7
            )
            lr_dict = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss" if SEPARATE_TRAIN_TEST else "train_loss",
            }

            return [optimizer], [lr_dict]
        else:
            return optimizer

    def _calc_batch_error(self, pred_coords, ref_coords):
        n_cols, n_rows = self.input_shape
        def calc_error(pred_coords, ref_coords):
            diff = pred_coords - ref_coords
            diff[0::2] *= 100 / n_cols
            diff[1::2] *= 100 / n_rows
            return (diff ** 2).mean()

        error = 0.0
        batch_size = len(ref_coords)
        for i in range(batch_size):
            err = calc_error(pred_coords[i], ref_coords[i])
            if err > 1e6:
              print(f"\nMAX_ERROR:\n{ref_coords[i]}\n{pred_coords[i]}")

            error += err

        return error / batch_size

    def validation_step(self, batch, batch_idx):
        images, ref_coords_batch = batch

        pred_coords_batch = self.model(images)

        loss = self.loss(pred_coords_batch, ref_coords_batch)
        error = self._calc_batch_error(pred_coords_batch, ref_coords_batch)
        lr = self.optimizers().param_groups[0]["lr"]

        metrics = {"val_loss": loss, "error": error, "lr": lr}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True,
                      logger=True)

        return metrics
    

def GetCheckpointsList(dirpath):
    monitor="error" if SEPARATE_TRAIN_TEST else "train_loss_epoch"    
    return ModelCheckpoint(
            dirpath=dirpath,
            filename=f"{{epoch}}-{{{monitor}:.3f}}",
            monitor=monitor,
            mode="min",
            save_top_k=1,
        )

def GetMyTransform():
    return A.Compose([
        A.RandomRotate90(always_apply=True),
        A.Rotate(),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3,
                                    contrast_limit=0.3, p=0.5),
            A.Blur(blur_limit=2, p=0.3)
        ], p=0.3),
        A.Equalize(p=0.2),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

@dataclass
class ModelParams:
    input_shape = (96, 96)
    ncpu = os.cpu_count()
    batch_size = 64
    lr = 2*1e-2
    is_full_load_on_start=True


def train_model(train_gt, train_img_dir, fast_train, dir_path):
    model_params = ModelParams()

    dataset_train = FacePointsDataset(
        mode="train", model_input_shape=model_params.input_shape,
        img_dir=train_img_dir, train_gt=train_gt,
        transform=GetMyTransform(),
        is_full_load_on_start=model_params.is_full_load_on_start,
        fraction_fast_train=0.02 if fast_train else None
    )

    if SEPARATE_TRAIN_TEST:
        dataset_val = FacePointsDataset(
            mode="val", model_input_shape=model_params.input_shape,
            img_dir=train_img_dir, train_gt=train_gt,
            is_full_load_on_start=model_params.is_full_load_on_start
        )

    datalaoder_train = DataLoader(
        dataset_train,
        batch_size=model_params.batch_size,
        shuffle=True,
        num_workers=model_params.ncpu,
    )

    if SEPARATE_TRAIN_TEST:
        datalaoder_val = DataLoader(
            dataset_val,
            batch_size=model_params.batch_size,
            shuffle=False,
            num_workers=model_params.ncpu
        )

    training_module = TrainingPointDetector(model_params.input_shape, lr=model_params.lr)

    if fast_train:
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False
        )
    else:
        trainer = pl.Trainer(
            max_epochs=150,
            accelerator="gpu",
            devices=1,
            callbacks=GetCheckpointsList(dir_path),
            log_every_n_steps=5,
            num_sanity_val_steps=0,
        )
        
    if SEPARATE_TRAIN_TEST:
        trainer.fit(training_module, datalaoder_train, datalaoder_val)
    else:
        trainer.fit(training_module, datalaoder_train)
    
    return trainer.model


def train_detector(train_gt, train_img_dir, fast_train=True): #TODO: fast_train
    dir_path = "runs/fp_detector"
    model = train_model(train_gt, train_img_dir, fast_train, dir_path)
    return model

def detect(model_filename, test_img_dir):
    model_params = ModelParams()
    
    dataset_val = FacePointsDataset(
        mode="inf",
        model_input_shape=model_params.input_shape,
        img_dir=test_img_dir,
        is_full_load_on_start=False
    )
    
    training_module = TrainingPointDetector.load_from_checkpoint(model_filename,
                                                                 input_shape=(96, 96))
    model = training_module.model
    model.eval()

    res = {}
    with torch.no_grad():
        size = len(dataset_val)
        for i in range(size):
            img, inv_point_coefs, name = dataset_val[i]
            y = model(img[None, ...])[0]

            y[0::2] *= inv_point_coefs[0]
            y[1::2] *= inv_point_coefs[1]
            
            res[name] = y.numpy()

            if i % (6000 // 10) == 0:
                print(f"{i:4} / {size}")
    
    return res