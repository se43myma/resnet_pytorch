import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.padding = 0
        if stride == 1:
            self.padding = 1
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, self.padding), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 1, padding=1), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        if stride != 1 and in_channels != out_channels:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride), nn.BatchNorm2d(out_channels))
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            ResBlock(64, 64, 1),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
            ResBlock(256, 512, 2),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, 2),
            nn.Sigmoid())

    def forward(self, x):
        out = self.resnet(x)
        return out

