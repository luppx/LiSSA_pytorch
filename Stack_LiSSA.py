# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 21:28:41 2020

@author: slu
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
import pandas as pd
import numpy as np


class AutoEncoder(nn.Module):
    def __init__(self, n_in, n_out, LR=1e-3):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(n_out, n_in), nn.ReLU())
        self.optimizer = optim.Adam(self.parameters(), lr=LR)

    def lgem_loss(self, labels, preds, delta_outputs):
        """
        :param labels: (batch_size, 1, input_dim)
        :param preds: (batch_size, 1, input_dim)
        :return:
        """
        mse = torch.mean(torch.pow((labels-preds), 2))
        preds = preds.unsqueeze(0)
        preds = preds.repeat(50, 1, 1, 1)
        ssm = torch.mean(torch.pow((preds-delta_outputs), 2))
        lgem = torch.mean(torch.pow((torch.sqrt(mse)+0.01*torch.sqrt(ssm)), 2))
        return lgem

    def forward(self, x):
        encode = self.encoder(x)

        if self.training:
            decode = self.decoder(encode)
            self.optimizer.zero_grad()
            # produce numbers in range[r1,r2) in pytorch, use r1+(r2-r1)*torch.rand(tensor_size)
            delta_outputs = []
            for i in range(H):
                temp = -Q + 2*Q*torch.rand(size=x.shape, dtype=torch.float)
                out_h = self.reconstruct(self.encoder((x + temp).float()))
                delta_outputs.append(out_h)
            delta_outputs = torch.stack(delta_outputs)
            loss = self.lgem_loss(x, decode, delta_outputs)
            loss.backward()
            self.optimizer.step()
            print('[Epoch %3d, Batch %3d] loss: %.5f' %
                  (epoch + 1, step + 1, loss.item()))
        return encode.detach() #RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed.

    def reconstruct(self, x):
        return self.decoder(x)


class StackAutoencoder(nn.Module):
    def __init__(self, n_in):
        super(StackAutoencoder, self).__init__()
        self.ae1 = AutoEncoder(n_in, 128, 1e-3)
        self.ae2 = AutoEncoder(128, 64, 1e-3)
        self.ae3 = AutoEncoder(64, 32, 1e-3)
        self.fc = nn.Linear(32, 1)
        self.fc_loss = nn.MSELoss()
        self.fc_optimizer = optim.Adam(self.parameters(), 3e-3)

    def forward(self, x, y):
        # change [batchsize, n1, n2] to [batchsize, n1*n2], which can meet nn.Linear's params:[in_features, out_features]
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        encode1 = self.ae1(x)
        encode2 = self.ae2(encode1)
        encode3 = self.ae3(encode2)
        product_predict = self.fc(encode3 * max)

        if self.training:
            fc_loss = self.fc_loss(y, product_predict)
            self.fc_optimizer.zero_grad()
            fc_loss.backward()
            self.fc_optimizer.step()
            print('[Epoch %3d, Batch %3d] loss: %.5f' %
                  (epoch + 1, step + 1, fc_loss.item()))

        return encode3, product_predict

    def reconstruct(self, x):
        x = x.view(x.size(0), -1)
        ae2_reconstruct = self.ae3.reconstruct(x)
        ae1_reconstruct = self.ae2.reconstruct(ae2_reconstruct)
        x_reconstruct = self.ae1.reconstruct(ae1_reconstruct)
        return x_reconstruct


# =============================================================================
# class lgem(nn.Module):
#     def __init__(self, H, Q):
#         super(lgem, self).__init__()
#         self.H = H
#         self.Q = Q
#     
#     def forward(self, datas, labels, preds):
#         """
#         :param labels: (batch_size, time_step, input_dim, 3)
#         :param preds: (batch_size, time_step, input_dim, 3)
#         :return:
#         """
#         # produce numbers in range[r1,r2) in pytorch, use r1+(r2-r1)*torch.rand(tensor_size)
#         delta_outputs = []
#         for _ in range(self.H):
#             temp = -self.Q + 2*self.Q*torch.rand(size=labels.shape, dtype=torch.float16)
#             out_h = model(datas + temp)
#             delta_outputs.append(out_h)
#         delta_outputs = torch.cat(delta_outputs, axis=-1)
#         mse = torch.mean(torch.pow((labels-preds), 2), dim=0)
#         preds = preds.unsqueeze(2)
#         delta_outputs = torch.reshape(delta_outputs, shape=[torch.shape(delta_outputs)[0], 
#                                             torch.shape(delta_outputs)[1], self.H, -1, 3])
#         ssm = torch.mean(torch.mean(torch.pow((preds-delta_outputs),2), dim=2), dim=0)
#         lgem = torch.mean(torch.pow((torch.sqrt(mse)+0.01*torch.sqrt(ssm)), 2))
#         
#         return lgem
# =============================================================================


class LoadData(Dataset):
    def __init__(self, datapath):
        self.data = pd.read_excel(datapath)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        in_data = self.data.iloc[index, 0:-2]
        product = self.data.iloc[index, -2]
        
        in_data = np.array([in_data])
        product = np.array([product])

        return in_data, product


if __name__ == '__main__':
    H = 50
    Q = 0.1
    n_in = 195
    n_out = 1
    LR = 1e-3
    datapath = './data0407.xlsx'
    frame = pd.read_excel(datapath, usecols="A:GM")
    max = float(frame.stack().max())

    data = LoadData(datapath)
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = random_split(data, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


    model = StackAutoencoder(n_in)
    print("Start training")
    for epoch in range(100):
        model.train(mode=True) #tells model that you are training the model.
        total_time = time.time()
        for step, (x, y) in enumerate(train_loader):
            x = x.float() / max  # Normalize input to 0-1
            y = y.float()
            model = model.float()
            encode, predict_product = model(x, y)

            # debug
            print("x max:%.5f,  min:%.5f" % (torch.max(x).item(), torch.min(x).item()))
            print("encode max:%.5f,  min:%.5f" % (torch.max(encode).item(), torch.min(encode).item()))
            print("y max:%.5f,  min:%.5f" % (torch.max(y).item(), torch.min(y).item()))
            print("output max:%.5f,  min:%.5f" % (torch.max(predict_product).item(), torch.min(predict_product).item()))

    print("Complete training")
