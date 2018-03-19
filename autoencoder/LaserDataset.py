import torch
from torch.utils.data.dataset import Dataset, TensorDataset
import numpy as np

class LaserDataset(TensorDataset):

    def __init__(self, folder_dataset, transform=None):
        self.transform = transform
        # Open and load text file including the whole training data
        self.poi_laser = np.genfromtxt(folder_dataset+'poi_laser.csv', delimiter=",", dtype=np.float32)/43.0
        self.rov_laser = np.genfromtxt(folder_dataset+'rov_laser.csv', delimiter=",", dtype=np.float32)/43.0


    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):

        poi = self.poi_laser[index,:]
        rov = self.rov_laser[index,:]

        laser = np.array([poi, rov])

        # Convert image and label to torch tensors
        return torch.from_numpy(np.asarray(laser))

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.poi_laser)

