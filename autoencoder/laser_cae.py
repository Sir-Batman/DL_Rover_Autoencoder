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

# FOLDER_DATASET = "./Track_1_Wheel_Test/"
# plt.ion()

model_file = './conv_autoencoder.pth'

class LaserDataset(TensorDataset):

    def __init__(self, folder_dataset="../build/Results/Multirover_experts/10/", transform=None):
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


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(

            nn.Conv1d(2, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool1d(3, stride=3), 

            nn.Conv1d(16, 1, 4, stride=2, padding=1),
            # nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.MaxPool1d(3, stride=3)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1, 16, 6, stride=6), 
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 2, 6, stride=6, padding=0), 
            nn.ReLU(True),
            # nn.ConvTranspose2d(8, 2, 2, stride=2, padding=1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        if self.training:
            return x2
        else:
            return x1, x2

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

def train():
    num_epochs = 500
    batch_size = 128
    learning_rate = 1e-2


    dataset = LaserDataset(folder_dataset="../build/Results/Multirover_experts/10/", transform=None)
    print("dataset of size: {}\n".format(len(dataset)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = AutoEncoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)
    model.train()
    for epoch in range(num_epochs):
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
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, num_epochs, loss.data[0]))
        # if epoch % 10 == 0:
        #     pic = to_img(output.cpu().data)
        #     save_image(pic, './dc_img/image_{}.png'.format(epoch))

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

    torch.save(model.state_dict(), model_file)
    print('saving model to file {}'.format(model_file))

def encode_data():
    num_epochs = 50
    batch_size = 128
    learning_rate = 1e-2


    dataset = LaserDataset(folder_dataset="../build/Results/Multirover_experts/0/", transform=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = AutoEncoder().cuda()


    print('Loading model parameters...')
    pretrained_dict = torch.load(os.path.join("./conv_autoencoder.pth"))
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
