import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import os
#import matplotlib.pyplot as plt
#import os

import pdb
from L1Penalty import L1Penalty

# FOLDER_DATASET = "./Track_1_Wheel_Test/"
# plt.ion()

class AutoEncoder(nn.Module):
    def __init__(self, l1weight=0.05):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(

            nn.Conv1d(2, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool1d(3, stride=3), 

            nn.Conv1d(32, 16, 5, stride=1, padding=2),
            # nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool1d(3, stride=3),          

            nn.Conv1d(16, 2, 5, stride=1, padding=2),
            # nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.MaxPool1d(5, stride=5)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(2, 16, 12, stride=4, padding=0), 
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 32, 3, stride=3, padding=0), 
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 2, 3, stride=3, padding=0), 
            nn.Sigmoid()
        )

        self.l1weight = l1weight

    def forward(self, x):
        x = self.encoder(x)

        x = L1Penalty.apply(x, self.l1weight)

        x = self.decoder(x)

        return x


    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

