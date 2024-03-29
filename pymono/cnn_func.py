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
     
#super().__init__() instead of super(CNN_basic, self).__init__()

# class MonoDataset(Dataset):
#     """
#     Loads the data to pytorch 
#     self.dataset ->[d1, d2...] where di ->[img (normalized), vector (x,y,z)]
#     variable xyze intoduced for backward compatibility.
#     Old csv files had xyz info, new csv files have xyze info. To use older files
#     or new files (ignoring e), set xyze = False, to use new files (adding e) 
#     xyze is also False. 
#     """

#     def __init__(self, data_path: str, frst_file: int, lst_file: int, 
#                  norm=False, resize=False, mean=None, std=None, size=None, xyze=False):
        
#         self.dataset = []
#         if norm and resize:
#             transf = v2.Compose([
#                             v2.Resize(size=size, antialias=True),
#                             v2.Normalize(mean=[mean], std=[std])])
                    
#         elif norm:
#             transf = v2.Normalize(mean=[mean], std=[std])
                    
#         elif resize:
#             transf = v2.Resize(size=size, antialias=True)
                   

#         img_name, lbl_name, indx = files_list_npy_csv(data_path)
#         print(f"Loading files with indexes: {indx[frst_file:lst_file]}")

#         for i in indx[frst_file:lst_file]:
#             images = np.load(f'{data_path}/{img_name}_{i}.npy')
#             metadata = pd.read_csv(f'{data_path}/{lbl_name}_{i}.csv')

#             for img, meta in zip(images, metadata.values):
#                 if xyze: # label contains xyze but we want to compare only with xyz
#                     ee = meta[1:-1] # energy in the csv in last position
#                 else:
#                     ee = meta[1:] # either old format (xyz) or new format (xyze)
#                 if norm or resize:
#                     self.dataset.append((transf(img), ee))
#                 else:
#                     self.dataset.append(((img, ee)))

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         image, position = self.dataset[idx]
#         image = torch.tensor(image, dtype=torch.float).unsqueeze(0) # Add channel dimension
#         position = torch.tensor(position, dtype=torch.float)

#         return image, position


# class MonoDatasetOld(Dataset):
#     """
#     Loads the data to pytorch 
#     self.dataset ->[d1, d2...] where di ->[img (normalized), vector (x,y,z)]
#     """

#     def __init__(self, data_path: str, frst_file: int, lst_file: int):
        
#         self.dataset = []
#         for i in range(frst_file, lst_file):
#             images = np.load(f'{data_path}/images_{i}.npy')
#             metadata = pd.read_csv(f'{data_path}/metadata_{i}.csv')

#             for img, meta in zip(images, metadata.values):
#                 #self.dataset.append((img/img.max(), meta[1:]))
#                 self.dataset.append((img, meta[1:]))

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         image, position = self.dataset[idx]
#         image = torch.tensor(image, dtype=torch.float).unsqueeze(0) # Add channel dimension
#         position = torch.tensor(position, dtype=torch.float)

#         return image, position


# def split_data(dataset, train_fraction=0.7, val_fraction=0.2):
#     """
#     Split the data between train, validation and test

#     """
#     train_size = int(train_fraction * len(dataset))  # training
#     val_size = int(val_fraction * len(dataset))    # validation
#     test_size = len(dataset) - train_size - val_size  # test
#     train_indices = range(train_size)
#     val_indices = range(train_size, train_size + val_size)
#     test_indices = range(train_size + val_size, len(dataset))

#     trsz = namedtuple('trsz',
#            'train_size, val_size, test_size, train_indices, val_indices, test_indices')
#     return trsz(train_size, val_size, test_size, train_indices, val_indices, test_indices)
    

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
    convolution (3x3) , reLU batch norm and MaxPool: (8,6,1) => (8,8,64)
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
        # convolution (3x3) , reLU batch norm and MaxPool: (16,16,1) => (8,8,64)
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

def single_run(train_loader, device, model, optimizer, criterion):
    print(f"** Run for 1 event**")

    for epoch in range(1):
        print(f"epoch = {epoch}")
    
        for i, (images, labels) in enumerate(train_loader):  
            if i>1: break
            print(f"i = {i}")
            print(f"images = {images.shape}")
            print(f"labels = {labels.shape}")
            images = images.to(device)
            labels = labels.to(device)
            
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)

            print(f"outputs = {outputs.data.shape}")
           
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            
            loss.backward()
            optimizer.step()
    
            print(f"Train Step {i + 1}/{len(train_loader)}, Loss: {loss.data.item()}")



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
    
def train_cnn(train_loader, val_loader, model, optimizer, device, criterion, 
              batch_size, epochs=10, iprnt=100):
    """
    train the CNN
    """

    print(f"Training with  ->{len(train_loader)*batch_size} images")
    print(f"size of train loader  ->{len(train_loader)} images")
    print(f"Evaluating with  ->{len(val_loader)*batch_size} images")
    print(f"size of eval loader  ->{len(val_loader)} images")
    print(f"Running for epochs ->{epochs}")

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        train_losses_epoch, val_losses_epoch = [], []

        #logging.debug(f"\nEPOCH {epoch}")
        print(f"\nEPOCH {epoch}")

        # Training step
        for i, (images, positions) in enumerate(train_loader):

            images = images.to(device)
            positions = positions.to(device)

            model.train()  #Sets the module in training mode.
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            outputs = model(images) # image --> model --> (x,y,z)
            loss = criterion(outputs, positions) # compare labels with predictions
            loss.backward()  # backward pass
            optimizer.step()
            
            train_losses_epoch.append(loss.data.item())
            if((i+1) % (iprnt) == 0):
                #logging.debug(f"Train Step {i + 1}/{len(train_loader)}, Loss: {loss.data.item()}")
                print(f"Train Step {i + 1}/{len(train_loader)}, Loss: {loss.data.item()}")

        #print(f"Done with training after epoch ->{epoch}")
        #print(f"Start validations in epoch ->{epoch}")
        
        # Validation step
        with torch.no_grad():  #gradients do not change
            model.eval()       # Sets the module in evaluation mode.
            
            for i, (images, positions) in enumerate(val_loader):

                images = images.to(device)
                positions = positions.to(device)

                outputs = model(images)
                loss = criterion(outputs, positions)

                val_losses_epoch.append(loss.data.item())
                if((i+1) % (iprnt) == 0):
                    #logging.debug(f"Validation Step {i + 1}/{len(val_loader)}, Loss: {loss.data.item()}")
                    print(f"Validation Step {i + 1}/{len(val_loader)}, Loss: {loss.data.item()}")

        #print(f"Done with validation after epoch ->{epoch}")
        train_losses.append(np.mean(train_losses_epoch))
        val_losses.append(np.mean(val_losses_epoch))
        print(f"--- EPOCH {epoch} AVG TRAIN LOSS: {np.mean(train_losses_epoch)}")
        print(f"--- EPOCH {epoch} AVG VAL LOSS: {np.mean(val_losses_epoch)}")
    
    #logging.info(f"Out of loop after epoch ->{epoch}")
    return train_losses, val_losses


def evaluate_cnn(test_loader, model, device, classical=False, pixel_size = 6, energy=False):
    
    true_x, true_y, true_z = [],[],[]
    if energy:
        true_e = []

    mean_x, mean_y = [],[]
    sigma_x, sigma_y = [],[]
    predicted_x, predicted_y, predicted_z = [],[],[]
    if energy:
        predicted_e = []
    with torch.no_grad():

        model.eval()
        for i, (images, positions) in enumerate(test_loader):

            images = images.to(device)
            outputs = model(images).cpu()

            for x in positions[:,0]: true_x.append(x)
            for y in positions[:,1]: true_y.append(y)
            for z in positions[:,2]: true_z.append(z)
            if energy:
                for e in positions[:,3]: true_e.append(e)

            for x in outputs[:,0]: predicted_x.append(x)
            for y in outputs[:,1]: predicted_y.append(y)
            for z in outputs[:,2]: predicted_z.append(z)
            if energy:
                for e in outputs[:,3]: predicted_e.append(e)

            if classical:
                for img in images.cpu().squeeze().numpy():
                    mu_x, mu_y, sd_x, sd_y = weighted_mean_and_sigma(img)
                    mean_x.append(mu_x); mean_y.append(mu_y)
                    sigma_x.append(sd_x); sigma_y.append(sd_y)

    # Convert to numpy arrays
    true_x = np.array(true_x); true_y = np.array(true_y); true_z = np.array(true_z)
    if energy:
        true_e = np.array(true_e)

    predicted_x = np.array(predicted_x) 
    predicted_y = np.array(predicted_y); predicted_z = np.array(predicted_z)
    if energy:
       predicted_e = np.array(predicted_e)

    mean_x = np.array(mean_x); mean_y = np.array(mean_y)
    sigma_x = np.array(sigma_x); sigma_y = np.array(sigma_y)

    # Compute deltas for the NN.
    delta_x_NN = true_x - predicted_x
    delta_y_NN = true_y - predicted_y
    delta_z_NN = true_z - predicted_z

    if energy:
        delta_e_NN = true_e - predicted_e

    # Compute deltas for the classical method
    if classical: 
        delta_x_classical = true_x - pixel_size*mean_x
        delta_y_classical = true_y - pixel_size*mean_y
    else:
        delta_x_classical = 0.0
        delta_y_classical = 0.0

    if energy:
        tdeltas = namedtuple('tdeltas',
            'delta_x_NN, delta_y_NN, delta_z_NN, delta_e_NN, delta_x_classical, delta_y_classical')
        return tdeltas(delta_x_NN, delta_y_NN, delta_z_NN, delta_e_NN, delta_x_classical, delta_y_classical)
    else:
        tdeltas = namedtuple('tdeltas',
            'delta_x_NN, delta_y_NN, delta_z_NN, delta_x_classical, delta_y_classical')
        return tdeltas(delta_x_NN, delta_y_NN, delta_z_NN, delta_x_classical, delta_y_classical)



    