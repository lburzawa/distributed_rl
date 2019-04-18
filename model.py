import numpy as np
import torch
from torch import nn
import torch.optim as optim
from math import ceil
import time
import copy
from torchvision.utils import save_image
import os
import shutil
import argparse

class DoomNet(nn.Module):
    def __init__(self, num_classes):
        super(DoomNet,self).__init__()
        self.relu = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride = 2, padding = 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size = 3, stride = 2, padding = 1)
        self.bn4 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 3 * 3, 128)
        self.fc_val = nn.Linear(128, 1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
   
        x = x.view(x.size(0), 32 * 3 * 3)
        x = self.relu(self.fc1(x))
        v=self.fc_val(x)
        y=self.fc(x)

        return (y, v)

