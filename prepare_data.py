import cv2, torch
import numpy as np
import matplotlib.pyplot as plt
from utils import params

class Data:
    def __init__(self, params, df, labels=None, transform=None, is_Train=True):
        self.df = df
        self.labels = labels
        self.transform = transform
        self.is_Train = is_Train
        self.params = params

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        file_name = self.df['id'].values[idx]
        if self.is_Train:
            file_path = './data/train-1/train-1/{}.png'.format(file_name)
        else:
            file_path = './data/test/test/{}.jpg'.format(file_name)
        image = cv2.imread(file_path)
        if len(image.shape)==2:
            image = np.stack((image, image, image), axis=-1)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        if self.is_Train:
            label = self.labels.values[idx]
            target = torch.zeros(params["nb_classes"])
            for cls in label.split():
                target[int(cls)] = 1

            return image, target

        return image