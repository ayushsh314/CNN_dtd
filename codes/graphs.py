# importing the libraries
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD


train_x = np.load("train_x.npy")
train_y = np.load("train_y.npy")

i = 0
plt.figure(figsize=(10,10))
plt.axis("off")
plt.subplot(221), plt.imshow(train_x[i]*255, cmap='gray')
plt.subplot(222), plt.imshow(train_x[i+25]*255, cmap='gray')
plt.subplot(223), plt.imshow(train_x[i+50]*255, cmap='gray')
plt.subplot(224), plt.imshow(train_x[i+275]*255, cmap='gray')
plt.show()

train_losses = np.load("train_losses.npy")
val_losses = np.load("val_losses.npy")

# plotting the training and validation loss
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()