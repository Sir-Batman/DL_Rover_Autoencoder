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

class LaserDataset(TensorDataset):

    def __init__(self, folder_dataset="../build/Results/Multirover_experts/0/", transform=None):
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


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(360, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Linear(64, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(True), 
            nn.Linear(12, 8))

        self.decoder = nn.Sequential(
            nn.Linear(8, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), 
            nn.Linear(128, 360), 
            nn.Tanh())

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        
        return x2

    def encode(self, x):
        return self.encoder(x)

    def decoder(self. x):
        return self.decoder(x)


num_epochs = 10
batch_size = 128
learning_rate = 1e-3


dataset = LaserDataset(folder_dataset="../build/Results/Multirover_experts/0/", transform=None)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

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

for data in dataloader:
    laser_data = data
    laser_data = Variable(laser_data).cuda
    encoded = autoencoder.encode(laser_data)
    decoded = autoencoder.decode(encoded)

    pdb.set_trace()

torch.save(model.state_dict(), './autoencoder.pth')



# if __name__ == '__main__':
# 	main()
