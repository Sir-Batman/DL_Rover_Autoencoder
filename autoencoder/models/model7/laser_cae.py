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

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(

            nn.Conv1d(2, 32, 5, stride=3, padding=1),
            nn.ReLU(True),
            # nn.MaxPool1d(3, stride=3), 

            nn.Conv1d(32, 16, 5, stride=3, padding=1),
            nn.ReLU(True),
            # nn.MaxPool1d(3, stride=3),          

            nn.Conv1d(16, 1, 5, stride=5, padding=1),
            nn.ReLU(True),
            # nn.MaxPool1d(5, stride=5)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1, 16, 5, stride=5, padding=0), 
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 32, 5, stride=3, padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 2, 5, stride=3, padding=1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2


    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

