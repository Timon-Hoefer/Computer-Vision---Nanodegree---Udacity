## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # (1,224,224)
        self.conv1 = nn.Conv2d(1, 32, 5)
        # (32, 220,220)
        self.pool1 = nn.MaxPool2d(2, 2)
        # (32, 110, 110)
        self.conv2 = nn.Conv2d(32, 64, 3)
        # (64, 108, 108)
        self.pool2 = nn.MaxPool2d(4,4)
        # (64, 27, 27)
        self.conv3 = nn.Conv2d(64, 128, 4)
        # (128, 24, 24)
        self.pool3 = nn.MaxPool2d(2,2)
        # (128, 12, 12)
        self.linear1 = nn.Linear(128*12*12, 1000)
        self.linear2 = nn.Linear(1000, 500)
        self.linear3 = nn.Linear(500, 136)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
       
        
        
        
                              
            
                               
                               
                               
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.dropout2(x)
        
        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.linear1(x))
        x = self.dropout3(x)
        x = F.relu(self.linear2(x))
        #x = self.dropout4(x)
        x = self.linear3(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
