"""MVTec Anomaly Detection dataset loader.

Expected folder structure:
    <root>/<classname>/train/good/
    <root>/<classname>/test/<anomaly_type>/
    <root>/<classname>/ground_truth/<anomaly_type>/
"""
import os
from enum import Enum

import PIL
import torch
from torchvision import transforms

_CLASSNAMES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill",
    "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MVTecDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for the MVTec-AD benchmark."""

    def __init__(
        self,
        source: str,
        classname: str,
        resize: int = 256,
        imagesize: int = 224,
        split: DatasetSplit = DatasetSplit.TRAIN,
        train_val_split: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            source: Root directory of the MVTec dataset.
            classname: Category name (e.g. ``'bottle'``).
            resize: Resize shorter side to this value before cropping.
            imagesize: Final centre-crop size.
            split: TRAIN / VAL / TEST.
            train_val_split: Fraction used for training when < 1.0.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ])
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN
        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx: int) -> dict:
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.shape[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.replace("\\", "/").split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self) -> int:
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in files
                ]

                if self.train_val_split < 1.0:
                    n = len(imgpaths_per_class[classname][anomaly])
                    split_idx = int(n * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = (
                            imgpaths_per_class[classname][anomaly][:split_idx]
                        )
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = (
                            imgpaths_per_class[classname][anomaly][split_idx:]
                        )

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anom_mask_path = os.path.join(maskpath, anomaly)
                    mask_files = sorted(os.listdir(anom_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anom_mask_path, x) for x in mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

        data_to_iterate = []
        for classname in sorted(imgpaths_per_class):
            for anomaly in sorted(imgpaths_per_class[classname]):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    mask = (
                        maskpaths_per_class[classname][anomaly][i]
                        if self.split == DatasetSplit.TEST and anomaly != "good"
                        else None
                    )
                    data_to_iterate.append([classname, anomaly, image_path, mask])

        return imgpaths_per_class, data_to_iterate
