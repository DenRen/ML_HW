# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import timm
from numpy.random import randint, choice

import os
from os.path import join
import csv
import json
import tqdm
import pickle
import typing
import cv2

from dataclasses import dataclass

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import KNeighborsClassifier

from run import calc_metric

CLASSES_CNT = 205

class DatasetRTSD(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения датасета.
    :param root_folders: список путей до папок с данными
    :param path_to_classes_json: путь до classes.json
    """
    def __init__(self, root_folders, path_to_classes_json,
                 model_input_shape=None, load_on_init=False) -> None:
        super(DatasetRTSD, self).__init__()
        self.load_on_init = load_on_init
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)

        # список пар (путь до картинки, индекс класса)
        self.samples = []
        for root_folder in root_folders:
            for class_name in os.listdir(root_folder):
                dir_path = join(root_folder, class_name)

                class_dirs = os.listdir(dir_path)
                for img_name in class_dirs:
                    path = join(dir_path, img_name)
                    self.samples.append((path, self.class_to_idx[class_name]))

        # cловарь из списков картинок для каждого класса,
        # classes_to_samples[индекс класса] = [список чисел-позиций картинок в self.samples]
        self.classes_to_samples = { self.class_to_idx[class_name] : []
                                    for class_name in self.classes }
        for idx, (path, class_idx) in enumerate(self.samples):
            self.classes_to_samples[class_idx].append(idx)

        #аугментации + нормализация + ToTensorV2
        self.transform =  A.Compose([
            # A.Normalize(), # In extern transform
            A.Resize(*model_input_shape) if model_input_shape else A.NoOp(),
            ToTensorV2()
        ])

        if self.load_on_init:
            self.imgs = []
            for path, _ in self.samples:
                self.imgs.append(self._open_img(path))

    @staticmethod
    def _open_img(path):
        with Image.open(path) as img_file:
            img = img_file.convert("RGB")
            img = np.array(img, dtype=np.float32)
            return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        path, class_idx = self.samples[index]
        if self.load_on_init:
            img = self.imgs[index]
        else:
            img = self._open_img(path)

        img = self.transform(image=img)["image"]

        return img, path, class_idx


    @staticmethod
    def get_classes(path_to_classes_json):
        """
        Считывает из classes.json информацию о классах.
        :param path_to_classes_json: путь до classes.json
        """
        with open(path_to_classes_json, "r") as file:
            data = json.load(file)

        class_to_idx = { key : value["id"] for key, value in data.items() }
        classes = sorted([ [v, k] for k, v in class_to_idx.items() ])
        classes = [ k for v, k in classes ]

        return classes, class_to_idx

class SubDatasetRTSD(Dataset):
    def __init__(self, dataset, full_ds:DatasetRTSD=None, transform=None):
        super().__init__()

        self.dataset = dataset
        self.transform = transform
        
        if full_ds is not None:
            self.classes_to_samples = { key : []
                                        for key in full_ds.classes_to_samples.keys() }
            
            for idx, (_, _, class_idx) in enumerate(dataset):
                self.classes_to_samples[class_idx].append(idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, path, class_idx = self.dataset[index]

        if self.transform is not None:
            image = image.permute(1, 2, 0).numpy().astype(np.uint8)
            image = self.transform(image=image)["image"].astype(np.float32)
            image = torch.from_numpy(image).permute(2, 0, 1)

        assert image.dtype == torch.float32

        return image, class_idx

class TestData(Dataset):
    """
    Класс для чтения и хранения тестового датасета.
    :param root: путь до папки с картинками знаков
    :param path_to_classes_json: путь до classes.json
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    def __init__(self, root, path_to_classes_json, annotations_file=None,
                 model_input_shape=None):
        super(TestData, self).__init__()
        self.root = root
        self.samples = sorted(os.listdir(root))
        # преобразования: ресайз + нормализация + ToTensorV2
        self.transform = A.Compose([
            A.Resize(*model_input_shape) if model_input_shape else A.NoOp(),
            A.Normalize(),
            ToTensorV2()
        ])
        self.targets = None
        if annotations_file is not None:
            with open(annotations_file, "r") as csv_file:
                path_to_class_name = { path : class_name
                                       for path, class_name in list(csv.reader(csv_file))[1:]}

            self.classes, self.class_to_idx = DatasetRTSD.get_classes(path_to_classes_json)
            # словарь, targets[путь до картинки] = индекс класса
            self.targets = { path : self.class_to_idx[path_to_class_name[path]]
                             for path in self.samples }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        path = self.samples[index]
        with Image.open(join(self.root, path)) as img_file:
            img = img_file.convert("RGB")

        img = np.array(img, dtype=np.float32)
        img = self.transform(image=img)["image"]

        if self.targets is not None and path in self.targets.keys():
            return img, path, self.targets[path]

        return img, path, -1

class RTSDDataModule(pl.LightningDataModule):
    def __init__(self,
                 full_ds,
                 model_input_shape=None,
                 batch_size: int = 32,
                 train_transform=None,
                 num_workers:int=1,
                 enable_custom_batch_sampler:bool=False):
        super().__init__()
        
        self.full_ds = full_ds
        self.model_input_shape = model_input_shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.enable_custom_batch_sampler = enable_custom_batch_sampler

    def setup(self, stage=None):
        assert stage in ["fit", "validate"]

        gen = torch.Generator().manual_seed(123)
        train_ds, val_ds = random_split(self.full_ds, [0.8, 0.2], generator=gen)
        # train_ds, val_ds, _ = random_split(self.full_ds, [0.04, 0.01, 0.95], generator=gen) #TODO: REMOVE
        
        self.train_ds = SubDatasetRTSD(
            train_ds,
            full_ds=self.full_ds if self.enable_custom_batch_sampler else None,
            transform=self.train_transform)
        self.val_ds = SubDatasetRTSD(val_ds, transform=A.Normalize())
        
        self.train_batch_sampler = None
        if self.enable_custom_batch_sampler:
            self.train_batch_sampler = CustomBatchSampler(self.train_ds, 8, 4)

    def _get_commom_dataloader(self, dataset, batch_sampler=None):
        return DataLoader(dataset,
                          batch_size=self.batch_size if batch_sampler is None else 1,
                          num_workers=self.num_workers,
                          batch_sampler=batch_sampler)

    def _set_train_transform(self, train_transform):
        self.train_transform = train_transform

    def train_dataloader(self):
        self.train_ds.transform = self.train_transform
        return self._get_commom_dataloader(self.train_ds, self.train_batch_sampler)

    def val_dataloader(self):
        return self._get_commom_dataloader(self.val_ds)

class CustomNetwork(pl.LightningModule):
    """
    Класс, реализующий нейросеть для классификации.
    :param features_criterion: loss-функция на признаки, извлекаемые нейросетью перед
                               классификацией (None когда нет такого лосса)
    :param internal_features: внутреннее число признаков
    """
    def __init__(self, features_criterion=None, internal_features=1024, pretrained=False,
                 patience=7):
        super(CustomNetwork, self).__init__()
        self.patience = patience
        self.model = timm.models.efficientnet.efficientnet_b2_pruned(pretrained=pretrained)

        self.features_criterion = features_criterion
        
        num_features = self.model.classifier.in_features
        if self.features_criterion is None:
            self.model.classifier = nn.Sequential(
                nn.Linear(num_features, internal_features),
                nn.SiLU(),
                nn.Linear(internal_features, CLASSES_CNT)
            )
        else:
            self.model.classifier = nn.Linear(num_features, internal_features)        

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
        logits = self.model(images)
        return F.log_softmax(logits, dim=1) if self.features_criterion is None else logits

    def predict(self, x):
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.
        :param x: батч с картинками
        """
        if self.features_criterion is None:
            return self.forward(x).argmax(dim=1)
        raise RuntimeError("Not implemented")

    def training_step(self, train_batch, batch_idx):
        img, class_idx = train_batch
        preds = self.forward(img)
        
        if self.features_criterion is None:
            loss = F.nll_loss(preds, class_idx)

            logs = {"train_loss": loss}

            acc = torch.sum(preds.argmax(axis=1) == class_idx) / class_idx.shape[0]
            self.log("train_acc", acc, on_step=True, prog_bar=True)
        else:
            loss = self.features_criterion(preds, class_idx)

        self.log("train_loss", loss, on_step=True, prog_bar=True)
        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, val_batch, batch_idx):
        img, class_idx = val_batch
        preds = self.forward(img)
        
        metrics = {}
        if self.features_criterion is None:
            loss = F.nll_loss(preds, class_idx)
            acc = torch.sum(preds.argmax(axis=1) == class_idx) / class_idx.shape[0]

            self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
            metrics["val_acc"] = acc
        else:
            loss = self.features_criterion(preds, class_idx)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        metrics["val_loss"] = loss
        
        if self.optimizers():
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr", lr, on_step=True, on_epoch=True, prog_bar=True)
            metrics["lr"] = lr

        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.2,
            patience=self.patience
        )
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"#"train_loss" if self.is_learn_on_val else "val_loss"
        }
        return [optimizer], [lr_dict]

def GetModelCheckpoint(train_on_val=False):
    return ModelCheckpoint(
        dirpath="runs/bird_classifier",
        filename="{epoch}-{val_loss:.3f}",#"{epoch}-{val_acc:.3f}" if not train_on_val else "{epoch}-{train_acc:.3f}",
        monitor="val_loss", #"val_acc" if not train_on_val else "train_acc",
        mode="min", #"max",
        every_n_epochs=1,
        enable_version_counter=True,
        save_on_train_epoch_end=True,
        save_last=True,
        save_top_k=2
    )

def learn_epochs(module, num_epoch, lr, unfreeze_layers, data_module,
                 check_val_every_n_epoch=1, profiler=None):
    module.lr = lr
    module.unfreeze_layers(unfreeze_layers)

    trainer = pl.Trainer(
        callbacks=[GetModelCheckpoint()],
        max_epochs=num_epoch,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=check_val_every_n_epoch,
        profiler=profiler,
    )
    trainer.fit(module, data_module)

@dataclass
class ModelParams:
    input_shape = (224, 224)
    num_cpu = os.cpu_count()
    batch_size = 32
    load_on_init=True
    num_classes = CLASSES_CNT

def train_simple_classifier():
    """Функция для обучения простого классификатора на исходных данных."""
       
    config = ModelParams()

    dataset = DatasetRTSD(
        root_folders=["cropped-train"],
        path_to_classes_json="classes.json",
        model_input_shape=config.input_shape,
        load_on_init=False)
        
    data_module = RTSDDataModule(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_cpu)

    MyTransform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.4,
                                       contrast_limit=0.3, p=0.8),
            A.Blur(blur_limit=7, p=0.3),
            A.CLAHE(),
        ], p=0.4),
        A.OneOf([
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30),
            A.ChannelDropout(),
            A.Equalize(p=0.1),
        ]),
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=30,
                            border_mode=1),
            A.Affine(mode=1),
        ]),
        A.GaussNoise(var_limit=(40, 400), p=0.3),
        A.CoarseDropout(min_holes=7, max_holes=50,
                        min_height=7, max_height=50,
                        min_width=3, max_width=7, p=0.2),
        A.Normalize(always_apply=True)
    ])
    
    data_module._set_train_transform(MyTransform)
    
    module = CustomNetwork()
    data_module.batch_size = 32
    learn_epochs(module, unfreeze_layers=5, num_epoch=30, lr=1e-3, data_module=data_module)
    
    return module

def apply_classifier(model, test_folder, path_to_classes_json):
    """
    Функция, которая применяет модель и получает её предсказания.
    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param path_to_classes_json: путь до файла с информацией о классах classes.json
    """
    
    config = ModelParams()
    test_data = TestData(test_folder, path_to_classes_json,
                         model_input_shape=config.input_shape)
    data_loader = DataLoader(test_data, batch_size=config.batch_size)
    classes = DatasetRTSD.get_classes(path_to_classes_json)[0]
    
    # список словарей вида {'filename': 'имя файла', 'class': 'строка-название класса'}
    results = []
    for batch in data_loader:
        imgs, paths, _ = batch
        preds = model.predict(imgs)
        results += [{ 'filename' : path, 'class' : classes[class_idx] }
                      for path, class_idx in zip(paths, preds) ]
        
        # break # TODO
    
    return results

def test_classifier(model, test_folder, path_to_classes_json, annotations_file):
    """
    Функция для тестирования качества модели.
    Возвращает точность на всех знаках, Recall на редких знаках и Recall на частых знаках.
    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    config = ModelParams()
    test_data = TestData(test_folder, path_to_classes_json, annotations_file,
                         model_input_shape=config.input_shape)
    data_loader = DataLoader(test_data, batch_size=config.batch_size)
    classes = DatasetRTSD.get_classes(path_to_classes_json)[0]
    
    with open(path_to_classes_json, "r") as file:
        classes_info = json.load(file)
    class_name_to_type = {k: v["type"] for k, v in classes_info.items()}
    
    # список словарей вида {'filename': 'имя файла', 'class': 'строка-название класса'}
    classes_trues, classes_preds = [], []
    for batch in data_loader:
        imgs, _paths, classes_true = batch
        classes_pred = model.predict(imgs)
        
        classes_trues += [ classes[v] for v in classes_true ]
        classes_preds += [ classes[class_idx] for class_idx in classes_pred ]
    
    total_acc = calc_metric(classes_trues, classes_preds, "all", class_name_to_type)   
    rare_recall = calc_metric(classes_trues, classes_preds, "rare", class_name_to_type) 
    freq_recall = calc_metric(classes_trues, classes_preds, "freq", class_name_to_type) 
    
    return total_acc, rare_recall, freq_recall

class SignGenerator(object):
    """
    Класс для генерации синтетических данных.
    :param background_path: путь до папки с изображениями фона
    """
    def __init__(self, background_path):
        self.background_path = background_path
        self.background_names = sorted(list(os.listdir(background_path)))

    def get_sample(self, icon):
        """
        Функция, встраивающая иконку на случайное изображение фона.
        :param icon: Массив с изображением иконки
        """
        
        hw_range = (16, 128+1)
        pad_range = (0, 15+1)
        
        h, w = randint(*hw_range), randint(*hw_range)
        icon = A.Compose([
            A.Resize(h, w, always_apply=True),
            A.CropAndPad(percent=randint(*pad_range) / 100, always_apply=True),
        ])(image=icon)["image"]
        
        icon[..., :3] = A.ColorJitter()(image=icon[..., :3])["image"]
        icon = A.Compose([
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
            A.MotionBlur(blur_limit=13, allow_shifted=False, always_apply=True),
            A.GaussianBlur(always_apply=True)
        ])(image=icon)["image"]
        
        bg_name = self.background_names[randint(0, len(self.background_names))]
        bg_path = join(self.background_path, bg_name)
        with Image.open(bg_path) as img_file:
            bg = np.array(img_file.convert("RGB"), dtype=np.uint8)
        
        bg = A.RandomCrop(h, w, always_apply=True)(image=bg)["image"]
        
        mask = (icon[..., 3] / 255)[..., None]
        res_fp32 = bg * (1 - mask) + icon[..., :3] * mask
        return res_fp32.astype(np.uint8)

def generate_one_icon(args):
    """
    Функция, генерирующая синтетические данные для одного класса.
    :param args: Это список параметров: [путь до файла с иконкой, путь до выходной папки,
                                         путь до папки с фонами, число примеров каждого класса]
    """
    icon_path, output_folder, background_path, samples_per_class = args
    gen = SignGenerator(background_path)
    
    class_name = icon_path.split("/")[-1].rsplit(".", 1)[0]
    imgs_dir = join(output_folder, class_name)
    
    if not os.path.exists(imgs_dir):
        os.mkdir(imgs_dir)
    
    for i in range(samples_per_class):
        with Image.open(icon_path) as img_file:
            icon = np.array(img_file.convert("RGBA"), dtype=np.uint8)
        sint_icon = gen.get_sample(icon)
        
        sint_icon_path = join(imgs_dir, f"{i:04}.png")
        Image.fromarray(sint_icon).save(sint_icon_path)    

def generate_all_data(output_folder, icons_path, background_path, samples_per_class = 1000):
    """
    Функция, генерирующая синтетические данные.
    Эта функция запускает пул параллельно работающих процессов, каждый из которых будет генерировать иконку своего типа.
    Это необходимо, так как процесс генерации очень долгий.
    Каждый процесс работает в функции generate_one_icon.
    :param output_folder: Путь до выходной директории
    :param icons_path: Путь до директории с иконками
    :param background_path: Путь до директории с картинками фона
    :param samples_per_class: Количество примеров каждого класса, которые надо сгенерировать
    """
    with ProcessPoolExecutor(8) as executor:
        params = [[join(icons_path, icon_file), output_folder, background_path, samples_per_class]
                  for icon_file in sorted(os.listdir(icons_path))]
        list(tqdm.tqdm(executor.map(generate_one_icon, params)))

def train_synt_classifier():
    """Функция для обучения простого классификатора на смеси исходных и ситетических данных."""
    config = ModelParams()

    dataset = DatasetRTSD(
        root_folders=["new_train"], # This must be generated by generate_all_data function
        path_to_classes_json="classes.json",
        model_input_shape=config.input_shape,
        load_on_init=False)
    
    data_module = RTSDDataModule(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_cpu)

    MyTransform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.4,
                                    contrast_limit=0.3, p=0.8),
            A.Blur(blur_limit=7, p=0.3),
            A.CLAHE(),
        ], p=0.4),
        A.OneOf([
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30),
            A.ChannelDropout(),
            A.Equalize(p=0.1),
        ]),
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=30,
                            border_mode=1),
            A.Affine(mode=1),
        ]),
        A.GaussNoise(var_limit=(40, 400), p=0.3),
        A.CoarseDropout(min_holes=7, max_holes=50,
                        min_height=7, max_height=50,
                        min_width=3, max_width=7, p=0.2),
        A.Normalize(always_apply=True)
    ])
    data_module._set_train_transform(MyTransform)
    
    module = CustomNetwork()
    children = list(module.model.children())
    for idx, child in enumerate(children):
        print(len(children) - idx, ": ", str(child).split("(")[0])
    
    data_module.batch_size = 32
    module.patience = 5
    learn_epochs(module, unfreeze_layers=5, num_epoch=30, lr=1e-3, data_module=data_module)
    
    return module

class FeaturesLoss(torch.nn.Module):
    """
    Класс для вычисления loss-функции на признаки предпоследнего слоя нейросети.
    """
    def __init__(self, margin: float) -> None:
        super(FeaturesLoss, self).__init__()
        self.margin = margin
        
    def forward(self, feats, class_idxs):
        pos, neg = 0.0, 0.0
        pos_num, neg_num = len(class_idxs), 0
        
        for i in range(0, len(class_idxs)):
            for j in range(i + 1, len(class_idxs)):
                val = (feats[i] - feats[j]).square().sum()
                if class_idxs[i] == class_idxs[j]:
                    pos += val
                    pos_num += 1
                else:
                    neg += F.relu(self.margin - val.sqrt()).square()
                    neg_num += 1
        
        loss = 0.5 * (pos / pos_num + neg / max(1, neg_num))
        # print(f"LOSS: {loss}")
        return loss
        
        # dist_pos = (feats[1] - feats[0]).square().sum(axis=-1)
        # dist_neg = F.relu(self.margin - dist_pos.sqrt()).square()
        # loss = 0.5 * torch.where(class_idxs[0] == class_idxs[1], dist_pos, dist_neg)
        # return loss.mean()

class CustomBatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    """
    Класс для семплирования батчей с контролируемым числом классов и примеров каждого класса.
    :param data_source: Это датасет RTSD
    :param elems_per_class: Число элементов каждого класса
    :param classes_per_batch: Количество различных классов в одном батче
    """
    def __init__(self, data_source: DatasetRTSD, elems_per_class, classes_per_batch):
        self.data_source = data_source
        self.elems_per_class = elems_per_class
        self.classes_per_batch = classes_per_batch

        self.num_classes = len(data_source.classes_to_samples.keys())
        
        min_num_examples = 10000000000
        for class_idx in data_source.classes_to_samples.keys():
            num_examples = len(data_source.classes_to_samples[class_idx])
            min_num_examples = min(num_examples, min_num_examples)
        
        self.total_elem_per_class = elems_per_class * (min_num_examples // elems_per_class)
        
    def __len__(self):
        batch_size = self.elems_per_class * self.classes_per_batch
        return (self.total_elem_per_class * self.num_classes + batch_size - 1) // batch_size
        
    def __iter__(self):
        num_groups = self.total_elem_per_class // self.elems_per_class
        
        batch_size = self.elems_per_class * self.classes_per_batch
        batches = []
        
        is_strange_mode = num_groups == 0
        
        for i_group in range(max(1, num_groups)):
            class_idxs = np.arange(self.num_classes)
            np.random.shuffle(class_idxs)
            
            batch = []
            for class_idx in class_idxs:
                pos_arr = self.data_source.classes_to_samples[class_idx]
                if is_strange_mode:
                    batch += choice(pos_arr, size=self.elems_per_class).tolist()
                else:
                    for i in range(self.elems_per_class):
                        pos = pos_arr[i_group * self.elems_per_class + i]
                        batch.append(pos)
                
                if len(batch) == batch_size:
                    batches.append(batch)
                    batch = []

            if len(batch) != 0:
                batches.append(batch)
                batch = []
    
        yield from batches

def train_better_model():
    """Функция для обучения классификатора на смеси исходных и ситетических данных с новым лоссом на признаки."""
    config = ModelParams()

    dataset = DatasetRTSD(
        root_folders=["new_train"], # This must be generated by generate_all_data function
        path_to_classes_json="classes.json",
        model_input_shape=config.input_shape,
        load_on_init=False)
    
    data_module = RTSDDataModule(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_cpu,
        #train_batch_sampler=CustomBatchSampler(dataset, 8, 8)
        )

    MyTransform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.4,
                                    contrast_limit=0.3, p=0.8),
            A.Blur(blur_limit=7, p=0.3),
            A.CLAHE(),
        ], p=0.4),
        A.OneOf([
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30),
            A.ChannelDropout(),
            A.Equalize(p=0.1),
        ]),
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=30,
                            border_mode=1),
            A.Affine(mode=1),
        ]),
        A.GaussNoise(var_limit=(40, 400), p=0.3),
        A.CoarseDropout(min_holes=7, max_holes=50,
                        min_height=7, max_height=50,
                        min_width=3, max_width=7, p=0.2),
        A.Normalize(always_apply=True)
    ])
    data_module._set_train_transform(MyTransform)
    
    module = CustomNetwork()
    children = list(module.model.children())
    for idx, child in enumerate(children):
        print(len(children) - idx, ": ", str(child).split("(")[0])
    
    data_module.batch_size = 32
    module.patience = 5
    learn_epochs(module, unfreeze_layers=5, num_epoch=30, lr=1e-3, data_module=data_module)
    
    return module

MY_TEST_ = False

class ModelWithHead:
    """
    Класс, реализующий модель с головой из kNN.
    :param n_neighbors: Количество соседей в методе ближайших соседей
    """
    def __init__(self, n_neighbors):
        if MY_TEST_:
            self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=1)
            self.emb_net = CustomNetwork(features_criterion=FeaturesLoss(2.0),
                                        internal_features=1024,
                                        pretrained=False)
        else:
            self.emb_net = CustomNetwork()
            
        self.emb_net.cpu().eval()

    def load_nn(self, nn_weights_path):
        """
        Функция, загружающая веса обученной нейросети.
        :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
        """
        self.emb_net.load_state_dict(torch.load(nn_weights_path, map_location='cpu'))
        self.emb_net.cpu().eval()

    def load_head(self, knn_path):
        """
        Функция, загружающая веса kNN (с помощью pickle).
        :param knn_path: Путь, откуда надо прочитать веса kNN
        """
        with open(knn_path, "rb") as bin:
            self.knn = pickle.load(bin)

    def predict(self, imgs):
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.
        :param imgs: батч с картинками
        """
        
        # предсказание нейросетевой модели
        with torch.no_grad():
            if MY_TEST_:
                features = self.emb_net.forward(imgs).detach().cpu().numpy()
                features = features / np.linalg.norm(features, axis=1)[:, None]
                
                # предсказание kNN на features
                # print(features)
                knn_pred = self.knn.predict(features)
                return knn_pred
            else:
                return self.emb_net.predict(imgs)

class IndexSampler(torch.utils.data.sampler.Sampler[int]):
    """
    Класс для семплирования батчей с картинками индекса.
    :param data_source: Это датасет RTSD с синтетическими примерами
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    def __init__(self, data_source, examples_per_class) -> None:
        self.data_source = data_source
        self.examples_per_class = examples_per_class
        self.num_classes = len(data_source.classes)
    
    def __len__(self):
        return self.num_classes * self.examples_per_class
    
    def __iter__(self):
        """Функция, которая будет генерировать список индексов элементов в батче."""
        res = []
        for class_idx in range(self.num_classes):
            res += choice(self.data_source.classes_to_samples[class_idx],
                          self.examples_per_class).tolist()
        yield from res


def calc_embs(emb_net, data_loader):
    emb_net.model.eval()
    with torch.no_grad():
        data_x, data_y = [], []
        for imgs, paths, class_idxs in tqdm.tqdm(data_loader):
            embs = emb_net.forward(imgs)
            data_x.append(embs.detach().cpu().numpy())
            data_y.append(class_idxs.detach().cpu().numpy())
            
        data_x = np.concatenate(data_x, axis=0)
        data_y = np.concatenate(data_y, axis=0)
    
        return data_x, data_y
    
def train_head(nn_weights_path, examples_per_class=20):
    """
    Функция для обучения kNN-головы классификатора.
    :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    config = ModelParams()
    
    model = ModelWithHead(examples_per_class)
    model.load_nn(nn_weights_path)
    
    dataset = DatasetRTSD(
        root_folders=["sint_icons"], # This must be generated by generate_all_data function
        path_to_classes_json="classes.json",
        model_input_shape=config.input_shape,
        load_on_init=False)
    
    data_loader = DataLoader(dataset, batch_size=32, num_workers=1,
                             sampler=IndexSampler(dataset, examples_per_class))
    
    data_x, data_y = calc_embs(model.emb_net, data_loader)
    data_x = data_x / np.linalg.norm(data_x, axis=1)[:, None]
    model.knn.fit(data_x, data_y)
    
    with open("knn_model.bin", "wb") as knn_file:
        pickle.dump(model.knn, knn_file)
    
