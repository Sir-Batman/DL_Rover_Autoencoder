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
from LaserDataset import LaserDataset

# The model number 
model_folder = 'model0'

model_file = 'models.'+ model_folder +'.laser_cae.AutoEncoder'

AutoEncoder = locate(model_file)
if AutoEncoder==None:
    raise TypeError, "Failed to find AutoEncoder"

autoencoder_param_file = 'models/'+model_folder+'/conv_autoencoder.pth'




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
    
    model = loadModel()

    # sample data
    poi_laser = np.genfromtxt(os.path.join('samples', 'train_poi_laser.csv'), delimiter=",", dtype=np.float32)/43.0
    rov_laser = np.genfromtxt(os.path.join('samples', 'train_rov_laser.csv'), delimiter=",", dtype=np.float32)/43.0

    # assume data holds your 2x360 NORMALIZED data with POI as first row, and ROV as second row
    # Note: numpy array need to have elements of dtype=np.float32
    data = np.array((poi_laser[0], rov_laser[0]))

    # change type of data to be of type np.float32 in order to play nice with weights in Conv1D
    data = data.astype(np.float32)

    # reshape the data so that it is in a batch size of 1
    data = data.reshape(1, 2, 360)

    # convert data to a tensor
    data_tensor = torch.from_numpy(np.asarray(data))

    # put tensor into variable
    laser_data = Variable(data_tensor).cuda()

    # encoded version of the laser data
    encoded = model.encode(laser_data)

    # decoded version of laser data
    decoded = model.decode(encoded)

    print("Initial Representation: ")
    print(laser_data)
    print("\nEncoded Representation: ")
    print(encoded)
    print("\n Decoded Representation: ")
    print(decoded)

if __name__ == '__main__':
    main()