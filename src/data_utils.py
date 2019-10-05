import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as albu


def get_image(img_path):
    """Use cv2 to load image
    Arg:
        img_path (str): the path to the image's location.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img / 255.


class HAMDataset(Dataset):
    def __init__(self, df, preprocess=None, transforms=None):
        self.df = df
        self.preprocess = preprocess
        self.transforms = transforms
    
    def __getitem__(self, idx):
        img_path = self.df['path'].iloc[idx]
        y = self.df['target'].iloc[idx]
        id_ = self.df['image_id'].iloc[idx]
        img = get_image(img_path)

        if self.preprocess:
            pre_process = self.preprocess(image=img)
            img = pre_process['image']

        if self.transforms:
            augment = self.transforms(image=img)
            img = augment['image']

        return img, y, id_
    
    def __len__(self):
        return len(self.df)


def to_tensor(x, **kwargs):
    return x.transpose(2,0,1).astype('float32')


def build_train_transform(brightness_limit=.2, 
                          contrast_limit=.2,
                          size=244):
    """Build argmentation process for training data
    Arg:
        brightness_limit (flaot): a upper and lower bound for brightening.
        contrast_limit (float): a upper and lower bound for contrast.
        size (int): the output size for images (3, size, size).
    """
    _transform = [
        albu.Resize(size, size),
        albu.HorizontalFlip(p=.5), 
        albu.VerticalFlip(p=.5), 
        albu.ShiftScaleRotate(p=.5),
        albu.RandomBrightnessContrast(brightness_limit, contrast_limit, p=.5),
        albu.Lambda(image=to_tensor),
        ]

    return albu.Compose(_transform)


def build_test_transform(size=244):
    """Build argmentation process for testing data
    Arg:
        size (int): the output size for images (3, size, size).
    """
    _transform = [
        albu.Resize(size, size),
        albu.Lambda(image=to_tensor),
        ]

    return albu.Compose(_transform)


def build_preprocess(mean, std):
    """Get corresponding preprocessing with pre-train weight
    Args:
        mean (list): The mean of RGB channel for pre-train weight.
        std (list): The std of RGB channel for pre-train weight.
    """
    _transform = [
        albu.Normalize(mean, std),
    ]

    return albu.Compose(_transform)


