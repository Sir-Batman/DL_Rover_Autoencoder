import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
# FOLDER_DATASET = "./Track_1_Wheel_Test/"
# plt.ion()

class LaserDataset(Dataset):

    def __init__(self, folder_dataset, transform=None):
        self.transform = transform
        # Open and load text file including the whole training data
        self.poi_laser = np.genfromtxt(folder_dataset+'poi_laser.csv', delimiter=",")
        self.rov_laser = np.genfromtxt(folder_dataset+'rov_laser.csv', delimiter=".")


    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):


        self.poi_laser[index,:]
        img = Image.open(self.__xs[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Convert image and label to torch tensors
        img = torch.from_numpy(np.asarray(img))
        label = torch.from_numpy(np.asarray(self.__ys[index]).reshape([1,1]))
        return img, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)