import torch.nn as nn

import sys 
import logging

logging.basicConfig(stream=sys.stdout,
                    format='%(levelname)s:%(message)s', level=logging.WARN)


     
class ResidualBlock(nn.Module):
    """
    Implements a residual block consisting in [Conv2d->BatchNorm2d->ReLU] + 
    [Conv2d->BatchNorm2d]. This residual is added to the input (then a second activation ReLU applied)
    
    If downsample = None (e.g, default first pass), then we obtain f(x) + x where 
    f(x) -> [Conv2d->BatchNorm2d->ReLU ->Conv2d->BatchNorm2d]. Otherwise the block is skipped. 
    
    """
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
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
        logging.warn(f"Residual Block: downsample ->{self.downsample}")
        logging.info(f"input: x ->{x.shape}")

        residual = x
        # This is the residual (in the case of no downsample)
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample:  # this is the residual in the case of downsample
            residual = self.downsample(x)
            
        out += residual # This is ist! f(x) + x 
        out = self.relu(out)

        return out
    

class ResNet34(nn.Module):
    """
    Implements t
    he Residual Network with 34 layers:
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
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet34, self).__init__()
        
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        self.info = True
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        logging.info(f" ## make_layer: planes = {planes},  blocks = {blocks}, stride = {stride}")
        logging.info(f" ## make_layer: in_planes={self.inplanes}")
        logging.info(f" ## make_layer: downsample = {downsample}")
        logging.info(f"layer block = 0: Block(in_channels={self.inplanes}, out_channels ={planes}, stride = {stride}, downsample = {downsample}")
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            logging.info(f" layer block = {i}: Block(in_channels={self.inplanes}, out_channels ={planes}, stride = 1, downsample = None")

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        if(self.info): logging.info(f" ResNet: input data shape =>{x.shape}")
            
        x = self.conv1(x)
        if(self.info): logging.info(f" ResNet: after conv1 =>{x.shape}")
            
        x = self.maxpool(x)
        if(self.info): logging.info(f" ResNet: after maxpool =>{x.shape}")
            
        x = self.layer0(x)
        if(self.info): logging.info(f" ResNet: after layer0 =>{x.shape}")
        
        x = self.layer1(x)
        if(self.info): logging.info(f" ResNet: after layer1 =>{x.shape}")
            
        x = self.layer2(x)
        if(self.info): logging.info(f" ResNet: after layer2 =>{x.shape}")
            
        x = self.layer3(x)
        if(self.info): logging.info(f" ResNet: after layer3 =>{x.shape}")
            
        x = self.avgpool(x)
        if(self.info): logging.info(f" ResNet: after avgpool =>{x.shape}")
            
        x = x.view(x.size(0), -1)
        if(self.info): logging.info(f" ResNet: after view =>{x.shape}")
            
        x = self.fc(x)
        if(self.info): logging.info(f" ResNet: after fc =>{x.shape}")

        self.info = False
        return x
