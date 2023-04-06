import numpy as np
import torch
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

train_x = np.load("train_x.npy")
train_y = np.load("train_y.npy")

# converting validation images into torch format
train_x = train_x.reshape(5076, 1, 128, 128)
train_x  = torch.from_numpy(train_x)

# converting the target into torch format
train_y = torch.from_numpy(train_y)

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
    
# Model class must be defined somewhere
model = Net()
model.load_state_dict(torch.load('codes/model.pth'))


def accuracy_score(arr1, arr2):
    accuracy = np.sum(np.equal(arr1,arr2))/len(arr1)
    return accuracy

# prediction for validation set
predictions = []
for i in range(5076) :
    with torch.no_grad():
        output = model(train_x[i:i+1])

    softmax = torch.exp(output)
    prob = list(softmax.detach().numpy())
    predictions.append(np.argmax(prob, axis=1).item())

# accuracy on validation set
print("Accuracy of model: ",accuracy_score(train_y.tolist(), predictions))
