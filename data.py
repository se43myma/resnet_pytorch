from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image


train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data:pd.DataFrame, mode:str) -> None:
        self.data_frame = data
        self.mode = mode
        self._transform_train = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean, train_std), tv.transforms.RandomHorizontalFlip(p=0.5), tv.transforms.RandomVerticalFlip(p=0.5)])
        self._transform_val = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean, train_std)])
   
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        sample = self.data_frame.iloc[index]
        img = imread(sample.filename)
        img = gray2rgb(img)
        if self.mode == 'train':
            img = self._transform_train(img)
        elif self.mode == 'val':
            img = self._transform_val(img)
        label = torch.tensor((sample.crack, sample.inactive))
        return img, label