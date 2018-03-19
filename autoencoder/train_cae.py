import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import os
from pydoc import locate

import pdb

from LaserDataset import LaserDataset

# The model folder containing the model (We assume that the model is in the /models folder, so omit that initial part)
model_folder = 'model2'

# Change the model_file to point to the autoencoder you want to use
# Format the string such that it references the AutoEncoder as a submodule
# Example: model_file = 'models.model0.laser_cae.AutoEncoder'
model_file = 'models.'+model_folder+'.laser_cae.AutoEncoder'

params_file = os.path.join('models', model_folder, 'conv_autoencoder_new.pth')
dataset_folder = "../build/Results/Multirover_experts/0/"
testset_folder = "../build/Results/Multirover_experts/0/"

AutoEncoder = locate(model_file)

if AutoEncoder==None:
    raise TypeError, "Failed to find AutoEncoder"


def train():
    print("Training Model from {}".format(model_file))
    num_epochs = 1000
    batch_size = 256
    learning_rate = 1e-3


    dataset = LaserDataset(folder_dataset=dataset_folder, transform=None)
    print("Training Data from {}".format(dataset_folder))    
    print("dataset of size: {}".format(len(dataset)))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Test Data from {}".format(testset_folder))    
    testset = LaserDataset(folder_dataset=testset_folder, transform=None) 

    print("testset of size: {}\n".format(len(testset)))

    testloader = DataLoader(testset, batch_size=len(testset), shuffle=False)

    model = AutoEncoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)
    
    for epoch in range(num_epochs):

        model.train()

        for data in dataloader:
            laser_data = data
            laser_data = Variable(laser_data).cuda()
            # ===================forward=====================
            output = model(laser_data)
            loss = criterion(output, laser_data)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], train loss:{:.4f}'
              .format(epoch+1, num_epochs, loss.data[0]))

        model.eval()

        for test in testloader:
            laser_data = test
            laser_data = Variable(laser_data).cuda()
            # ===================forward=====================
            output = model(laser_data)
            loss = criterion(output, laser_data)

        # ===================log========================
        print('epoch [{}/{}], test loss:{:.4f}'
              .format(epoch+1, num_epochs, loss.data[0]))

    # Verify what the encoded and decoded versions look like
    # pdb.set_trace()
    # model.eval()
    # for data in dataloader:
    #     laser_data = data
    #     laser_data = Variable(laser_data).cuda()
    #     encoded = model.encode(laser_data)
    #     decoded = model.decode(encoded)

    #     # x1, x2 = model(laser_data)
    #     pdb.set_trace()

    torch.save(model.state_dict(), params_file)
    print('saved model to file {}'.format(params_file))
    print('We would recommend you rename this model at asap to avoid overwriting.')

def encode_data():
    batch_size = 128

    dataset = LaserDataset(folder_dataset=testset_folder, transform=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = AutoEncoder().cuda()


    print('Loading model parameters...')
    pretrained_dict = torch.load(os.path.join(params_file))
    model_dict = model.state_dict()
    for key, value in pretrained_dict.iteritems():
        model_dict[key] = value
    model.load_state_dict(model_dict)

    # Here is some example code on how you would want to input the data
    # and run only the forward pass
    for data in dataloader:
        laser_data = data
        laser_data = Variable(laser_data).cuda()
        encoded = model.encode(laser_data)
        decoded = model.decode(encoded)


    # torch.save(model.state_dict(), './conv_autoencoder_.pth')

def main():
    # Run train() to create the autoencoder
    train()

    # Run encode_data() to use a prexisting autoencoder
    # encode_data()



if __name__ == '__main__':
	main()
