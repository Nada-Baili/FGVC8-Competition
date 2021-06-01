import albumentations.augmentations.functional as F
from albumentations import Compose, Normalize, Resize, HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate, IAAAdditiveGaussianNoise, RandomCrop
from albumentations.pytorch import ToTensorV2

from utils import *

class RandomCropIfNeeded(RandomCrop):
    def __init__(self, height, width, always_apply=False, p=1.0):
        super(RandomCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width

    def apply(self, img, h_start=0, w_start=0, **params):
        h, w, _ = img.shape
        return F.random_crop(img, min(self.height, h), min(self.width, w), h_start, w_start)

def get_transforms(*, data):
    assert data in ('train', 'valid')

    if data == 'train':
        return Compose([
            Resize(params["SIZE"], params["SIZE"]),
            RandomCropIfNeeded(params["SIZE"]*2, params["SIZE"]*2),
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(p=0.3),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=15, p=0.3),
            IAAAdditiveGaussianNoise(p=0.3),

            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(params["SIZE"], params["SIZE"]),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])