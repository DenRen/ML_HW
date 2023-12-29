from torch.utils.data import random_split, Dataset, DataLoader
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import albumentations as A

import numpy as np
import cv2

import os
from pathlib import Path
from PIL import Image

from dataclasses import dataclass

def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            filename, class_id = line.rstrip('\n').split(',')
            res[filename] = int(class_id)
    return res

def OpenImg(path, dtype) -> np.ndarray:
    with Image.open(path) as img_file:
        img = img_file.convert("RGB")
        arr = np.array(img, dtype=dtype)
    return arr

class BirdsDatasetStorage():
    def __init__(
        self,
        mode: str,
        model_input_shape,
        img_dir: Path,
        class_idxs,
        is_full_load_on_start: bool,
        use_part:float=1.0
    ):
        """
        This dataset is special for MIPT CV course testing system
        mode may be "train" or "inference"
        If mode is "train", then split all data in "train" dir to fit and val
        """

        assert mode in ["train", "inference"]

        self.model_input_shape = model_input_shape

        self.img_dir = Path(img_dir)
        self.img_names = sorted(os.listdir(self.img_dir))
        self.class_idxs = class_idxs
        
        if use_part < 1.0:
            last_idx = max(0, int(len(self.img_names) * use_part) - 1)
            self.img_names = self.img_names[:last_idx]
            
            keys = list(self.class_idxs.keys())[:last_idx]
            self.class_idxs_ = {}
            for key in keys:
                self.class_idxs_[key] = self.class_idxs[key]
            self.class_idxs = self.class_idxs_

        self.is_full_load_on_start = is_full_load_on_start
        if self.is_full_load_on_start:        
            num_imgs = len(self.img_names)
            h, w, c = *self.model_input_shape, 3
            self.imgs = np.zeros((num_imgs, h, w, c), dtype=np.float32)

            for i in range(num_imgs):
                self.imgs[i] = self._open_img_to_numpy(i)

    def _open_img_to_numpy(self, idx):
        img_path = self.img_dir / self.img_names[idx]
        image = OpenImg(img_path, np.float32)

        # Strange resize, todo: optimize
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = torchvision.transforms.functional.resize(
            image, self.model_input_shape, antialias=True,
        )
        return image.permute(1, 2, 0).numpy()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        if self.is_full_load_on_start:
            image = self.imgs[index]
        else:
            image = self._open_img_to_numpy(index)
        
        if self.class_idxs is None:
            return image, self.img_names[index]
        
        class_idx = self.class_idxs[self.img_names[index]]
        return image, class_idx

class BirdsDataset(Dataset):
    def __init__(self, dataset_storage, transform=None):
        self.storage = dataset_storage
        self.transform = transform

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, index):
        image, class_idx = self.storage[index]

        if self.transform is not None:
            image = image.astype(np.uint8)
            image = self.transform(image=image)["image"].astype(np.float32)

        assert image.dtype == np.float32

        image = torch.from_numpy(image).permute(2, 0, 1)
        return image, class_idx

class BirdsDatasetInference(Dataset):
    def __init__(self, dataset_storage):
        self.storage = dataset_storage

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, index):
        image, name = self.storage[index]
        assert image.dtype == np.float32

        image = torch.from_numpy(image).permute(2, 0, 1)
        return image, name

class BirdsDataModule(pl.LightningDataModule):
    def __init__(self,
                 img_dir: Path,
                 class_idxs,
                 batch_size: int = 32,
                 train_transform=None,
                 mode:str="train",
                 model_input_shape=None,
                 num_workers:int=1,
                 is_full_load_on_start:bool=True,
                 use_part:float=1.0):
        super().__init__()
        self.train_transform = train_transform
        self.mode = mode
        self.model_input_shape = model_input_shape
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.storage = BirdsDatasetStorage(self.mode, self.model_input_shape,
                                           img_dir, class_idxs, is_full_load_on_start,
                                           use_part)

    def _setup_train(self, stage):
        gen = torch.Generator().manual_seed(123)
        train_storage, val_storage = random_split(self.storage, [0.8, 0.2],
                                                  generator=gen)
        self.train_ds = BirdsDataset(train_storage, self.train_transform)
        self.val_ds = BirdsDataset(val_storage)

    def _setup_inference(self, stage):
        self.ds = BirdsDatasetInference(self.storage)

    def setup(self, stage=None):
        if self.mode == "train":
            assert stage in ["fit", "validate"]
        match self.mode:
            case "train":
                self._setup_train(stage)
            case "inference":
                self._setup_inference(stage)

    def _get_commom_dataloader(self, dataset):
        return DataLoader(dataset,
                          batch_size=self.batch_size)

    def _set_train_transform(self, train_transform):
        self.train_transform = train_transform

    def train_dataloader(self):
        self.train_ds.transform = self.train_transform
        return self._get_commom_dataloader(self.train_ds)

    def val_dataloader(self):
        return self._get_commom_dataloader(self.val_ds)
    
    def predict_dataloader(self): # Only mode == "inference"
        return self._get_commom_dataloader(self.ds)

class CustomClassifier(pl.LightningModule):
    def __init__(self, num_classes: int=50, pretrained:bool=True):
        super().__init__()

        self.num_classes = num_classes
        weights = torchvision.models.RegNet_X_3_2GF_Weights.IMAGENET1K_V2 if pretrained else None
        self.model = torchvision.models.regnet_x_3_2gf(weights)

        linear_size = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(linear_size),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=linear_size, out_features=1024, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=1024, out_features=self.num_classes, bias=False)
        )

        children = list(self.model.children())
        for child in children[:-1]:
            for param in child.parameters():
                param.requires_grad = False

    def unfreeze_layers(self, layers):
        children = list(self.model.children())

        if type(layers) == int:
            unfreeze_children = children[-layers:] if layers > 0 else []
        elif layers == "full":
            unfreeze_children = children[:]
        else:
            raise "Error layers"

        for child in unfreeze_children:
            for param in child.parameters():
                param.requires_grad = True

    def forward(self, images):
        return F.log_softmax(self.model(images), dim=1)

class BirdClassifier(pl.LightningModule):
    def __init__(self, lr=0.02, pretrained:bool=True):
        super(BirdClassifier, self).__init__()

        self.model = CustomClassifier(pretrained=pretrained)
        self.lr = lr
        self.lr_scheduler_enable = True

    def unfreeze_layers(self, layers):
        self.model.unfreeze_layers(layers)

    def forward(self, images):
        return self.model(images)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        img, class_idx = train_batch
        logits = self.forward(img)
        loss = self.cross_entropy_loss(logits, class_idx)

        logs = {"train_loss": loss}

        acc = torch.sum(logits.argmax(axis=1) == class_idx) / class_idx.shape[0]
        self.log("train_acc", acc, on_step=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, prog_bar=True)

        return {"loss": loss, "log": logs}

    def validation_step(self, val_batch, batch_idx):
        img, class_idx = val_batch
        logits = self.forward(img)
        loss = self.cross_entropy_loss(logits, class_idx)
        acc = torch.sum(logits.argmax(axis=1) == class_idx) / class_idx.shape[0]

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        metrics = {"val_loss": loss, "val_acc": acc}

        if self.optimizers():
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr", lr, on_step=True, on_epoch=True, prog_bar=True)
            metrics["lr"] = lr

        return metrics

    def test_step(self, val_batch, batch_idx):
        img, class_idx = val_batch
        logits = self.forward(img)
        loss = self.cross_entropy_loss(logits, class_idx)
        acc = torch.sum(logits.argmax(axis=1) == class_idx) / class_idx.shape[0]

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if not self.lr_scheduler_enable:
            return [optimizer]

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.2,
            patience=4
        )
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"
        }
        return [optimizer], [lr_dict]

def GetModelCheckpoint():
    return ModelCheckpoint(
        dirpath="runs/bird_classifier",
        filename="{epoch}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1
    )

def GetMyTransform():
    return A.Compose([
        A.Flip(),
        A.Rotate(),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3,
                                    contrast_limit=0.3, p=0.8),
            A.Blur(blur_limit=5, p=0.3),
            A.CLAHE(),
        ], p=0.4),
        A.Equalize(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30),
        A.RGBShift(r_shift_limit=127, g_shift_limit=127, b_shift_limit=127),
        A.OneOf([
            A.ColorJitter(),
            A.InvertImg(),
            A.ChannelDropout(),
            A.ChannelShuffle()
        ], p=0.4),
        A.OneOf([
            A.GaussNoise(var_limit=(40, 500), p=0.3),
            A.CoarseDropout(min_holes=7, max_holes=32,
                            min_height=7, max_height=16,
                            min_width=7, max_width=65)
        ], p=0.4),
        A.Affine(mode=cv2.BORDER_CONSTANT)
    ])

@dataclass
class ModelParams:
    input_shape = (224, 224)
    num_cpu = os.cpu_count()
    batch_size = 64
    lr = 2*1e-2
    num_classes = 50

def train_classifier(train_gt, train_img_dir, fast_train=True):
    config = ModelParams()
    data_module = BirdsDataModule(
        img_dir=train_img_dir,
        class_idxs=train_gt,
        batch_size = 32,
        train_transform=None,
        mode="train",
        model_input_shape=config.input_shape,
        num_workers=config.num_cpu,
        is_full_load_on_start=not fast_train,
        use_part=0.02
    )
    data_module._set_train_transform(GetMyTransform())

    checkpoint_callback = GetModelCheckpoint()
    def learn_epochs(model, num_epoch, lr, lr_sched:bool, unfreeze_layers,
                     check_val_every_n_epoch=1, profiler=None):
        model.lr = lr
        model.lr_scheduler_enable = lr_sched
        model.unfreeze_layers(unfreeze_layers)
        
        if fast_train:
            trainer = pl.Trainer(
                max_epochs=1,
                num_sanity_val_steps=0,
                logger=False,
                enable_checkpointing=False
            )
        else:
            trainer = pl.Trainer(
                callbacks=[checkpoint_callback],
                max_epochs=num_epoch,
                num_sanity_val_steps=2,
                check_val_every_n_epoch=check_val_every_n_epoch,
                profiler=profiler,
            )
        
        trainer.fit(model, data_module)

    def load_best_model(path=checkpoint_callback.best_model_path):
        return BirdClassifier.load_from_checkpoint(path)

    model = BirdClassifier(pretrained=False)
    learn_epochs(model, unfreeze_layers=1, num_epoch=50, lr=3e-2, lr_sched=True)
    return model

def classify(model_path, test_img_dir):
    config = ModelParams()
    data_module = BirdsDataModule(
        img_dir=test_img_dir,
        class_idxs=None,
        batch_size = 32,
        train_transform=None,
        mode="inference",
        model_input_shape=config.input_shape,
        num_workers=config.num_cpu,
        is_full_load_on_start=False
    )
    data_module.setup()
    data_loader = data_module.predict_dataloader()
    
    with open(model_path, "rb") as file:
        model = CustomClassifier(pretrained=False)
        model.load_state_dict(torch.load(file, map_location="cpu"))

    model.eval()
    res = {}
    with torch.no_grad():
        for batch in data_loader:
            imgs, names = batch
            class_prob = model(imgs)
            class_idxs = torch.argmax(class_prob, dim=1)
            for name, class_idx in zip(names, class_idxs):
                res[name] = class_idx
            
    return res
            