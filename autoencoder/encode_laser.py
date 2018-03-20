# This is a sample file to demonstrate how to use the developed autoencoder in laser_cae.py to encode data

import comms
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
import json

# The model folder (ex. `model0`)
model_folder = 'model1'

# construct the submodule path to the autoencoder
model_file = 'models.'+ model_folder +'.laser_cae.AutoEncoder'
AutoEncoder = locate(model_file)
if AutoEncoder==None:
    raise TypeError, "Failed to find AutoEncoder"

# construct path to autoencoder parameters
autoencoder_param_file = 'models/'+model_folder+'/conv_autoencoder_3layer.pth'


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

model = None
def forward(laser_data):
    global model
    if not model:
        model = loadModel()

    # assume data holds your 2x360 NORMALIZED data with POI as first row, and ROV as second row
    # Note: numpy array need to have elements of dtype=np.float32
    data = np.array(laser_data)
    #print "Data shape: ", data.shape
    #data = laser_data

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
    return encoded


if __name__ == '__main__':
    import time
    sent=0
    recv=0
    while True:
        # Get the input data
        data = comms.recieve("../topy")
        #time.sleep(0.1)
        #print "POI: ", POI_data
        #ROV_data = comms.recieve("../topy")
        recv+=1
        #print "recv: " , recv, "sent", sent
        data = json.loads(data)
        POI_data = data[0]
        ROV_data = data[1]
        laser_data = np.array([POI_data, ROV_data])
        laser_data = laser_data/43.0 #hard coded value
        #print laser_data
        # TODO Probably need to extract it to an array here
        #print "Received laser_data: ", POI_data, ROV_data
        encoded = forward((POI_data, ROV_data))
        output = ""
        for i in encoded.data.cpu().numpy()[0][0]:
            output += str(i) + ","
        # TODO need to format this encoded data?
        #print output
        comms.send(output, "../tocpp")
        sent+= 1
        #print "recv: " , recv, "sent", sent

