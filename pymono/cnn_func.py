import torch 
import torch.nn as nn
import numpy as np 
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch.nn.functional as F
import sys 

from collections import namedtuple
from pymono.aux_func import weighted_mean_and_sigma, files_list_npy_csv
import logging

logging.basicConfig(stream=sys.stdout,
                    format='%(levelname)s:%(message)s', level=logging.DEBUG)

def test_loggin():
     logging.debug(f"Hellow World")
     

class CNN_basic(nn.Module):
    """
    Defines a convolutional network with a basic architecture:
    convolution (3x3) , reLU batch norm and MaxPool: (8,8,1) => (4,4,128)
    convolution (2x2) , reLU batch norm and MaxPool: (4,4,128) => (2,2,256)
    convolution (2x2) , reLU batch norm and MaxPool: (2,2,256) => (1,1,512)
    drop (optional) 
    linear layer 512 => 3

    Input to the network are the pixels of the pictures, output (x,y,z)

    """
    def __init__(self, chi=128, dropout=False, dropout_fraction=0.2, energy=False):
        super().__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(1, chi, 3, padding=1) 
        self.bn1   = nn.BatchNorm2d(chi)
        self.conv2 = nn.Conv2d(chi, chi*2, 2, padding=1)
        self.bn2   = nn.BatchNorm2d(chi*2)
        self.conv3 = nn.Conv2d(chi*2, chi*4, 2, padding=1)
        self.bn3   = nn.BatchNorm2d(chi*4)
        self.pool = nn.MaxPool2d(2, 2)
        if energy:
            self.fc0 = nn.Linear(chi*4, 4) # add energy output.
        else:
            self.fc0 = nn.Linear(chi*4, 3)
        self.drop1 = nn.Dropout(p=dropout_fraction)
        self.debug = True

 
    def forward(self, x):

        if(self.debug): print(f"input data shape =>{x.shape}")
        # convolution (3x3) , reLU batch norm and MaxPool: (8,8,1) => (4,4,128)
        x = self.pool(self.bn1(F.leaky_relu(self.conv1(x))))
        
        if(self.debug): print(f"(8,8,1) => (4,4,128) =>{x.shape}")
        # convolution (2x2) , reLU batch norm and MaxPool: (4,4,128) => (2,2,256)
        x = self.pool(self.bn2(F.leaky_relu(self.conv2(x))))
        
        if(self.debug): print(f"(4,4,128) => (2,2,256) =>{x.shape}")
        # convolution (2x2) , reLU batch norm and MaxPool: (2,2,256) => (1,1,512)
        x = self.pool(self.bn3(F.leaky_relu(self.conv3(x))))
        
        if(self.debug): print(f"(2,2,256) => (1,1,512) =>{x.shape}")
        x = x.flatten(start_dim=1)
        # Flatten
        
        if(self.debug): print(f"(1,1,512) => (1,1,3) =>{x.shape}")
        
        if self.dropout: x = self.drop1(x)  # drop
        
        x = self.fc0(x)    # linear layer 512 => 3 (4)
        
        if(self.debug): print(x.shape)

        self.debug = False

        return x


class CNN_3x3(nn.Module):
    """
    Defines a convolutional network with a basic architecture:
    convolution (3x3) , reLU batch norm and MaxPool: (8,8,1) => (8,8,64)
    convolution (2x2) , reLU batch norm and MaxPool: (8,8,64) => (4,4,128)
    convolution (2x2) , reLU batch norm and MaxPool: (4,4,128) => (2,2,256)
    convolution (1x1) , reLU batch norm and MaxPool: (2,2,256) => (1,1,512)
    drop (optional) 
    linear layer 512 => 3

    Input to the network are the pixels of the pictures, output (x,y,z)

    """
    def __init__(self, chi=64, dropout=False, dropout_fraction=0.2, energy=False):
        super().__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(1, chi, 3, padding=1) 
        self.bn1   = nn.BatchNorm2d(chi)
        self.conv2 = nn.Conv2d(chi, chi*2, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(chi*2)
        self.conv3 = nn.Conv2d(chi*2, chi*4, 2, padding=1)
        self.bn3   = nn.BatchNorm2d(chi*4)
        self.conv4 = nn.Conv2d(chi*4, chi*8, 2, padding=1)
        self.bn4   = nn.BatchNorm2d(chi*8)
        self.pool = nn.MaxPool2d(2, 2)
        if energy:
            self.fc0 = nn.Linear(chi*8, 4) # add output for energy
        else:
            self.fc0 = nn.Linear(chi*8, 3)
        self.drop1 = nn.Dropout(p=dropout_fraction)
        self.debug = True

    def forward(self, x):

        if(self.debug): print(f"input data shape =>{x.shape}")
        # convolution (3x3) , reLU batch norm and MaxPool: (8,8,1) => (8,8,64)
        if(self.debug): print(f"(8,8,1) => (8,8,64) =>{x.shape}")
        x = self.pool(self.bn1(F.leaky_relu(self.conv1(x))))

        if(self.debug): print(f"(8,8,64) => (4,4,128) =>{x.shape}")
        # convolution (2x2) , reLU batch norm and MaxPool: (8,8,64) => (4,4,128)
        x = self.pool(self.bn2(F.leaky_relu(self.conv2(x))))
        
        if(self.debug): print(f"(4,4,128) => (2,2,256) =>{x.shape}")
        # convolution (2x2) , reLU batch norm and MaxPool: (4,4,128) => (2,2,256)
        x = self.pool(self.bn3(F.leaky_relu(self.conv3(x))))
        
        if(self.debug): print(f"(2,2,256) => (1,1,512) =>{x.shape}")
        # convolution (2x2) , reLU batch norm and MaxPool: (2,2,256) => (1,1,512)
        x = self.pool(self.bn4(F.leaky_relu(self.conv4(x))))
        
        #if(self.debug): print(f"(2,2,256) => (1,1,512) =>{x.shape}")
        if(self.debug): print(f"(1,1,512) => (1,1,3) =>{x.shape}")
        x = x.flatten(start_dim=1)
        # Flatten
        
        if self.dropout: x = self.drop1(x)  # drop
        
        x = self.fc0(x)    # linear layer 512 => 3
        
        if(self.debug): print(x.shape)

        self.debug = False

        return x
    

class ResBlock(nn.Module):
    """
    Implements a residual block consisting in [Conv2d->BatchNorm2d->ReLU] + 
    [Conv2d->BatchNorm2d]. This residual is added to the input (then a second activation ReLU applied)
    
    If downsample = None (e.g, default first pass), then we obtain f(x) + x where 
    f(x) -> [Conv2d->BatchNorm2d->ReLU ->Conv2d->BatchNorm2d]. Otherwise the block is skipped. 
    
    """
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, 
                                  stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                                  stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x # This is the residual (in the case of no downsample)
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample:  # this is the residual in the case of downsample
            residual = self.downsample(x)
            
        out += residual # This is it! f(x) + x 
        out = self.relu(out)

        return out



class ResNet(nn.Module):
    """
    Implements the Residual Network with 34 layers:
    The architecture is like this:
    1. Image (assumed 224 x 224) passes through a convolution (kernel 7x7) 
    with stride = 2 and padding = 3which increases the features from 3 to 64 and 
    reduces spatial dimensiones from 224 to (224 - 7 -2*3 +1)/2 =112, then batch normalization, 
    activation and MaxPool2d which further reduces dimensions to 56.
    2. The layer architecture is as follows (with a skip connection between each pair of layers) 
        6 layers of convolution 3x3 with 64 features
        8 layers of convolution 3x3 with 128 features (max pool 56 -> 28)
        12 layers of convolution 3x3 with 256 features (max pool 28 -> 24)
        6 layers of convolution 3x3 with 512 features (max pool 14 -> 7)
    3. Then avgpool and fc.
    
    """
    def __init__(self, block, layers, num_classes = 3, debug=False):
        super(ResNet, self).__init__()
        self.debug = debug
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        if self.debug:
            print(f" ## make_layer: planes = {planes},  blocks = {blocks}, stride = {stride}")
            print(f" ## make_layer: in_planes={self.inplanes}")
            print(f" ## make_layer: downsample = {downsample}")
            print(f"layer block = 0: Block(in_channels={self.inplanes}, out_channels ={planes}, stride = {stride}, downsample = {downsample}")
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            if self.debug:
                print(f" layer block = {i}: Block(in_channels={self.inplanes}, out_channels ={planes}, stride = 1, downsample = None")

        return nn.Sequential(*layers)
    
    def forward(self, x):
        if(self.debug): print(f" ResNet: input data shape =>{x.shape}")
            
        x = self.conv1(x)
        if(self.debug): print(f" ResNet: after conv1 =>{x.shape}")
            
        x = self.maxpool(x)
        if(self.debug): print(f" ResNet: after maxpool =>{x.shape}")
            
        x = self.layer0(x)
        if(self.debug): print(f" ResNet: after layer0 =>{x.shape}")
        
        x = self.layer1(x)
        if(self.debug): print(f" ResNet: after layer1 =>{x.shape}")
            
        x = self.layer2(x)
        if(self.debug): print(f" ResNet: after layer2 =>{x.shape}")
            
        x = self.layer3(x)
        if(self.debug): print(f" ResNet: after layer3 =>{x.shape}")
            
        x = self.avgpool(x)
        if(self.debug): print(f" ResNet: after avgpool =>{x.shape}")

        if(self.debug): print(f"(1,1,512) => (1,1,3) =>{x.shape}")
        x = x.flatten(start_dim=1)
        #x = x.view(x.size(0), -1)
        if(self.debug): print(f" ResNet: after flatten =>{x.shape}")
            
        x = self.fc(x)
        if(self.debug): print(f" ResNet: after fc =>{x.shape}")

        self.debug = False
        return x
        
class ResNet10(nn.Module):
    """
    Implements the Residual Network with 34 layers:
    The architecture is like this:
    1. Image passes through a convolution (kernel 3x3) 
    with stride = 1 and padding = 1 which increases the features from 3 to 64 and 
    preserves spatian dimensions, then batch normalization and activation. 
    # (W,H,1) => (W,H,64)
    
    2. The layer architecture is as follows (with a skip connection between each pair of layers) 
        6 layers of convolution 3x3 with 64 features
        8 layers of convolution 3x3 with 128 features (max pool 56 -> 28)
        12 layers of convolution 3x3 with 256 features (max pool 28 -> 24)
        6 layers of convolution 3x3 with 512 features (max pool 14 -> 7)
    3. Then avgpool and fc.
    
    """
    def __init__(self, block, num_classes = 3, dropout=False, dropout_fraction=0.2, debug=False):
        super(ResNet10, self).__init__()
        self.debug = debug
        self.inplanes = 64

        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(2, 2)
        self.layer0 = self._make_layer(block, 64, 1, stride = 1, nlyr = 1)
        self.layer1 = self._make_layer(block, 128, 1, stride = 2, nlyr = 2)
        self.layer2 = self._make_layer(block, 256, 1, stride = 2, nlyr = 3)
        self.layer3 = self._make_layer(block, 512, 1, stride = 2, nlyr = 4)
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(512, num_classes)
        self.dropout = dropout
        self.drop1 = nn.Dropout(p=dropout_fraction)

           
    def _make_layer(self, block, planes, blocks, stride, nlyr):
        downsample = None
        
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        if self.debug:
            print(f" ## make_layer {nlyr}: planes = {planes},  blocks = {blocks}, stride = {stride}")
            print(f" ## make_layer: in_planes={self.inplanes}")
            print(f" ## make_layer: downsample = {downsample}")
            #print(f"layer block = 0: Block(in_channels={self.inplanes}, out_channels ={planes}, stride = {stride}, downsample = {downsample}")
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            if self.debug:
                print(f" layer block = {i}: Block(in_channels={self.inplanes}, out_channels ={planes}, stride = 1, downsample = None")

        return nn.Sequential(*layers)
    
    def forward(self, x):
        if(self.debug): print(f" ResNet10: input data shape =>{x.shape}")
            
        x = self.conv1(x)
        if(self.debug): print(f" ResNet10: after conv1 =>{x.shape}")
            
        #x = self.maxpool(x)
        #if(self.debug): print(f" ResNet: after maxpool =>{x.shape}")
            
        x = self.layer0(x)
        if(self.debug): print(f" ResNet10: after layer0 =>{x.shape}")
        
        x = self.layer1(x)
        if(self.debug): print(f" ResNet10: after layer1 =>{x.shape}")
            
        x = self.layer2(x)
        if(self.debug): print(f" ResNet10: after layer2 =>{x.shape}")
            
        x = self.layer3(x)
        if(self.debug): print(f" ResNet10: after layer3 =>{x.shape}")
            
        x = self.avgpool(x)
        if(self.debug): print(f" ResNet10: after avgpool =>{x.shape}")

        x = x.flatten(start_dim=1)
        #x = x.view(x.size(0), -1)
        if(self.debug): print(f" ResNet10: after flatten =>{x.shape}")
        
        if self.dropout: x = self.drop1(x)  # drop
        x = self.fc(x)
        if(self.debug): print(f" ResNet10: after fc =>{x.shape}")

        self.debug = False
        return x
  
class FF(nn.Module):
    """
    A simple feed forward layer with 4 inputs and one hidden layer
    
    """
    def __init__(self, dropout=False, dropout_fraction=0.2):
        # call constructor from superclass
        super().__init__()

        self.dropout = dropout
        self.drop1 = nn.Dropout(p=dropout_fraction)
        # define network layers
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        #self.fc3 = nn.Linear(4, 2)
        self.fc4 = nn.Linear(4, 1)
        
    def forward(self, x):
        # define forward pass
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        #x = torch.sigmoid(self.fc3(x))
        if self.dropout: x = self.drop1(x)
        x = torch.sigmoid(self.fc4(x))
        return x
    
