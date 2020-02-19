from torchvision import transforms
import torch
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset
from enum import Enum, unique
import json
import os
import random

BASE_IMAGE_DIR = "/home/pdoyle/ssd/datasets/plant_disease/PlantVillage"
STDDEV = [0.192966, 0.170431, 0.207222]
MEAN = [0.458809, 0.475045, 0.411325]
VECTOR_SIZE = (200, 200)
CLASS_MAP = [
    "tomato_healthy",  # 0
    "tomato_early_blight",  # 1
    "bell_pepper_healthy",  # 2
    "potato_late_blight",  # 3
    "tomato_yellow_leaf_curl_virus",  # 4
    "potato_healthy",  # 5
    "tomato_leaf_mold",  # 6
    "bell_pepper_bacterial_spot",  # 7
    "tomato_septoria_leaf_spot",  # 8
    "potato_early_blight",  # 9
    "tomato_mosaic_virus",  # 10
    "tomato_late_blight",  # 11
    "tomato_bacterial_spot",  # 12
    "tomato_two_spotted_spidermite",  # 13
    "tomato_target_spot",  # 14
    ]


@unique
class SetType(Enum):
    test = "test"
    train = "train"
    val = "val"


def train_composer(stddev, mean):
    return transforms.Compose(
        [
            transforms.Resize(VECTOR_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
            ),
            transforms.ToTensor(),
            transforms.Normalize(stddev, mean),
        ]
    )


def inference_composer(stddev, mean):
    return transforms.Compose(
        [transforms.Resize(VECTOR_SIZE), transforms.ToTensor(), transforms.Normalize(stddev, mean)]
    )


class PlantDiseaseDataset(Dataset):
    def __init__(self, set_type: SetType, shuffle: bool = False):
        self.set_type = set_type
        self.shuffle = shuffle
        self.image_dir = BASE_IMAGE_DIR
        self.metadata = self._load_metadata()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, dict]:
        path = os.path.join(
            self.image_dir, self.metadata[idx]["class"], self.metadata[idx]["image"]
        )
        image = Image.open(path)
        if self.set_type == SetType.train:
            composer = train_composer(STDDEV, MEAN)
        else:
            composer = inference_composer(STDDEV, MEAN)
        return composer(image), self.metadata[idx]["class_num"], self.metadata[idx]

    def _load_metadata(self) -> List[dict]:
        with open("metadata.json") as f:
            metadata = json.load(f)
        ds = [doc for doc in metadata if doc["dataset"] == self.set_type.name]
        if self.shuffle:
            random.shuffle(ds)
        return ds
