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
    def __init__(self, l1weight=1e-5):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(360, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Linear(64, 12),
            # nn.BatchNorm1d(12),
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
        self.l1weight = l1weight

    def forward(self, x):
        x1 = self.encoder(x)
        x1 = L1Penalty.apply(x1, self.l1weight)

        x2 = self.decoder(x1)
        return x2

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


# num_epochs = 10
# batch_size = 128
# learning_rate = 1e-3


# dataset = LaserDataset(folder_dataset="../build/Results/Multirover_experts/0/", transform=None)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# model = autoencoder().cuda()
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
#                              weight_decay=1e-5)

# for epoch in range(num_epochs):
#     for data in dataloader:
#         laser_data = data
#         laser_data = Variable(laser_data).cuda()
#         # ===================forward=====================
#         output = model(laser_data)
#         loss = criterion(output, laser_data)
#         # ===================backward====================
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     # ===================log========================
#     print('epoch [{}/{}], loss:{:.4f}'
#           .format(epoch+1, num_epochs, loss.data[0]))

#     # if epoch % 10 == 0:
#     #     pic = to_img(output.cpu().data)
#     #     save_image(pic, './dc_img/image_{}.png'.format(epoch))

# for data in dataloader:
#     laser_data = data
#     laser_data = Variable(laser_data).cuda
#     encoded = autoencoder.encode(laser_data)
#     decoded = autoencoder.decode(encoded)

#     pdb.set_trace()

# torch.save(model.state_dict(), './autoencoder.pth')



# if __name__ == '__main__':
# 	main()
