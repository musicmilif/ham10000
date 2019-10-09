import random
import numpy as np
import cv2
from PIL import Image
from .auto_augmentation import ImageNetPolicy, SVHNPolicy, CIFAR10Policy
from albumentations import ImageOnlyTransform
import albumentations.augmentations.functional as F

__all__ = ["AutoAugmentWrapper", "RandomCropThenScaleToOriginalSize"]


class AutoAugmentWrapper(ImageOnlyTransform):
    def __init__(self, p=1.0):
        super(AutoAugmentWrapper, self).__init__(p)
        self.autoaugment = ImageNetPolicy()

    def apply(self, img, **params):
        img = Image.fromarray(img)
        img = self.autoaugment(img)
        img = np.asarray(img)
        return img
