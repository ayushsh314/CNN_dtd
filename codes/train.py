# importing the libraries
# import pandas as pd
import numpy as np

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

import os
from skimage.transform import resize
from skimage.color import rgb2gray

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# If not done previously--> Run below commented codes
1
# train_img = []
# train_y = []
# path = 'C:/Users/varun/Coding/pattern_cnn/dtd/images/'
# i=0
# for filename in os.listdir(path):
#     for images in os.listdir(path+filename) :
#         img = imread(path+filename+'/'+images)
#         img = rgb2gray(img)
#         img = resize(img, (128, 128, 1))
#         img = img.astype('float32')
#         img /= 255.0
#         train_img.append(img)
#         train_y.append(i)
#     i += 1

# train_x = np.array(train_img)

# train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1)
# val_y = np.array(val_y)
# train_y = np.array(train_y)

# train_x = np.save("train_x.npy", train_x)
# train_y = np.save("train_y.npy", train_y)
# val_x = np.save("val_x.npy", val_x)
# val_y = np.save("val_y.npy", val_y)

train_x = np.load("train_x.npy")
train_y = np.load("train_y.npy")
val_x = np.load("val_x.npy")
val_y = np.load("val_y.npy")

# converting training images into torch format
train_x = train_x.reshape(5076, 1, 128, 128)
train_x  = torch.from_numpy(train_x)

# converting the target into torch format
train_y = train_y.astype(int)
train_y = torch.from_numpy(train_y)

# converting validation images into torch format
val_x = val_x.reshape(564, 1, 128, 128)
val_x  = torch.from_numpy(val_x)

# converting the target into torch format
val_y = val_y.astype(int)
val_y = torch.from_numpy(val_y)

# Defining the model
class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(1, 32, 2),
            BatchNorm2d(32),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(0.4),

            Conv2d(32, 64, 2),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(0.4),
        )

        self.linear_layers = Sequential(
            Linear(64*31*31, 1024),
            Linear(1024, 47)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
    

# defining the model
model = Net()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.0005)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    
print(model)

# defining the number of epochs
print("Number of epochs to run: ")
n_epochs = int(input())
# empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []
# training the model
print("Training started")

for epoch in range(n_epochs):
    model.train()
    tr_loss = 0
    loss_train = 0
    loss_val = 0

    for i in range(5):
        # getting the training set
        x_train, y_train = Variable(train_x[i:i+1]), Variable(train_y[i:i+1]).type(torch.LongTensor)
        
        # converting the data into GPU format
        if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()
        
        # prediction for training set
        output_train = model(x_train)

        if torch.cuda.is_available():
            output_train = output_train.cuda()
            
        # computing the training loss
        loss = criterion(output_train, y_train)
        loss_train += loss.item()
        loss.backward()
        optimizer.step()

    print("Training done!")
        
    for i in range(5):
        # # prediction for validation set
        x_val, y_val = Variable(val_x[i:i+1]), Variable(val_y[i:i+1]).type(torch.LongTensor)
        if torch.cuda.is_available():
            x_val = x_val.cuda()
            y_val = y_val.cuda()
        
        output_val = model(x_val)
        loss_val += criterion(output_val, y_val).item()
    
    # Converting loss to tensor format
    # loss_train = Variable(torch.tensor((loss_train/324),dtype=torch.float32) , requires_grad=True)
    
    print("Validation done!")    
    # Appending for plotting graph
    val_losses.append(loss_val/564)
    train_losses.append(loss_train/5076)

    writer.add_scalar("Loss/train", loss_train/5076, epoch)
    print('Epoch : ',epoch+1, '\t', 'val loss :',loss_val/564, '\t', 'train loss :',loss_train/5076)
    
writer.flush()
writer.close()
np.save("train_losses.npy", train_losses)
np.save("val_losses.npy", val_losses)

torch.save(model.state_dict(), "C:/Users/varun/Coding/pattern_cnn/codes/model.pth")
print("Training done!!")