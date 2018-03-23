# This is a sample file to demonstrate how to use the developed autoencoder in laser_cae.py to encode data

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import os
import pdb
from pydoc import locate


# The model folder (ex. `model0`)
model_folder = 'model4'

# construct the submodule path to the autoencoder
model_file = 'models.'+ model_folder +'.laser_ae.AutoEncoder'
AutoEncoder = locate(model_file)
if AutoEncoder==None:
    raise TypeError, "Failed to find AutoEncoder"

# construct path to autoencoder parameters
autoencoder_param_file = 'models/'+model_folder+'/autoencoder_new.pth'


class LaserDataset(TensorDataset):

    def __init__(self, folder_dataset, transform=None):
        self.transform = transform
        # Open and load text file including the whole training data
        self.poi_laser = np.genfromtxt(folder_dataset+'poi_laser.csv', delimiter=",", dtype=np.float32)
        # self.rov_laser = np.genfromtxt(folder_dataset+'rov_laser.csv', delimiter=",", dtype=np.float32)


    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):

        poi = self.poi_laser[index,:]/43.0
        # rov = self.rov_laser[index,:]

        # laser = np.concatenate((poi, rov), axis=0)

        # Convert image and label to torch tensors
        # return torch.from_numpy(np.asarray(laser))
        return torch.from_numpy(np.asarray(poi))

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.poi_laser)


def loadModel():
    print("Training Model from {}".format(model_file))
    print("Using Model parameters from {}".format(autoencoder_param_file))
    # create model
    model = AutoEncoder().cuda()

    # load model parameters into dictionary
    print('Loading model parameters...')
    pretrained_dict = torch.load(autoencoder_param_file)
    model_dict = model.state_dict()
    for key, value in pretrained_dict.iteritems():
        model_dict[key] = value
    model.load_state_dict(model_dict)

    return model

def main():
    # pdb.set_trace()
    model = loadModel()
    model.eval()
    # sample data
    poi_laser = np.genfromtxt(os.path.join('samples', 'sample_poi_laser.csv'), delimiter=",", dtype=np.float32)/43.0
    # rov_laser = np.genfromtxt(os.path.join('samples', 'sample_rov_laser.csv'), delimiter=",", dtype=np.float32)/43.0
    # poi_laser[poi_laser==1.0] = 0.0
    # rov_laser[rov_laser==1.0] = 0.0

    # assume data holds your 2x360 NORMALIZED data with POI as first row, and ROV as second row
    # Note: numpy array need to have elements of dtype=np.float32
    # data = np.array((poi_laser[0], rov_laser[0]))
    data = np.array(poi_laser[0])

    # change type of data to be of type np.float32 in order to play nice with weights in Conv1D
    # data = data.astype(np.float32)

    # reshape the data so that it is in a batch size of 1
    data = data.reshape(1, 1, 360)

    # convert data to a tensor
    data_tensor = torch.from_numpy(np.asarray(data))

    # put tensor into variable
    laser_data = Variable(data_tensor).cuda()

    # encoded version of the laser data
    encoded = model.encode(laser_data)

    # decoded version of laser data
    decoded = model.decode(encoded)

    diff = decoded - laser_data

    print("Initial Representation: ")
    print(laser_data)
    print("\nEncoded Representation: ")
    print(encoded)
    print("\n Decoded Representation: ")
    print(decoded)
    print("\n Diff: ")
    print(diff)

if __name__ == '__main__':
    main()