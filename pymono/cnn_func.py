import torch 
import torch.nn as nn
import numpy as np 
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys 

from collections import namedtuple
from pymono.aux_func import weighted_mean_and_sigma
import logging

logging.basicConfig(stream=sys.stdout,
                    format='%(levelname)s:%(message)s', level=logging.DEBUG)

def test_loggin():
     logging.debug(f"Hellow World")
     
#super().__init__() instead of super(CNN_basic, self).__init__()

class MonoDataset(Dataset):
    """
    Loads the data to pytorch 
    self.dataset ->[d1, d2...] where di ->[img (normalized), vector (x,y,z)]
    """

    def __init__(self, data_path: str, frst_file: int, lst_file: int):
        
        self.dataset = []
        for i in range(frst_file, lst_file):
            images = np.load(f'{data_path}/images_{i}.npy')
            metadata = pd.read_csv(f'{data_path}/metadata_{i}.csv')

            for img, meta in zip(images, metadata.values):
                self.dataset.append((img/img.max(), meta[1:]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, position = self.dataset[idx]
        image = torch.tensor(image, dtype=torch.float).unsqueeze(0) # Add channel dimension
        position = torch.tensor(position, dtype=torch.float)

        return image, position


def split_data(dataset, train_fraction=0.7, val_fraction=0.2):
    """
    Split the data between train, validation and test

    """
    train_size = int(train_fraction * len(dataset))  # training
    val_size = int(val_fraction * len(dataset))    # validation
    test_size = len(dataset) - train_size - val_size  # test
    train_indices = range(train_size)
    val_indices = range(train_size, train_size + val_size)
    test_indices = range(train_size + val_size, len(dataset))

    trsz = namedtuple('trsz',
           'train_size, val_size, test_size, train_indices, val_indices, test_indices')
    return trsz(train_size, val_size, test_size, train_indices, val_indices, test_indices)
    

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
    def __init__(self, chi=128, dropout=False, dropout_fraction=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(1, chi, 3, padding=1) 
        self.bn1   = nn.BatchNorm2d(chi)
        self.conv2 = nn.Conv2d(chi, chi*2, 2, padding=1)
        self.bn2   = nn.BatchNorm2d(chi*2)
        self.conv3 = nn.Conv2d(chi*2, chi*4, 2, padding=1)
        self.bn3   = nn.BatchNorm2d(chi*4)
        self.pool = nn.MaxPool2d(2, 2)
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
        
        x = self.fc0(x)    # linear layer 512 => 3
        
        if(self.debug): print(x.shape)

        self.debug = False

        return x


class CNN_3x3(nn.Module):
    """
    Defines a convolutional network with a basic architecture:
    convolution (3x3) , reLU batch norm and MaxPool: (16,16,1) => (8,8,64)
    convolution (2x2) , reLU batch norm and MaxPool: (8,8,64) => (4,4,128)
    convolution (2x2) , reLU batch norm and MaxPool: (4,4,128) => (2,2,256)
    convolution (1x1) , reLU batch norm and MaxPool: (2,2,256) => (1,1,512)
    drop (optional) 
    linear layer 512 => 3

    Input to the network are the pixels of the pictures, output (x,y,z)

    """
    def __init__(self, chi=64, dropout=False, dropout_fraction=0.2):
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
        self.fc0 = nn.Linear(chi*8, 3)
        self.drop1 = nn.Dropout(p=dropout_fraction)
        self.debug = True

 
    def forward(self, x):

        if(self.debug): print(f"input data shape =>{x.shape}")
        # convolution (3x3) , reLU batch norm and MaxPool: (16,16,1) => (8,8,64)
        if(self.debug): print(f"(16,16,1) => (8,8,64) =>{x.shape}")
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
    
def train_cnn(train_loader, val_loader, model, optimizer, device, criterion, epochs=100):
    """
    train the CNN
    """
    train_losses, val_losses = [], []
    print(f"Running for epochs ->{epochs}")
    for epoch in range(epochs):
        train_losses_epoch, val_losses_epoch = [], []

        logging.debug(f"\nEPOCH {epoch}")
        #print(f"\nEPOCH {epoch}")

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
            if((i+1) % (len(train_loader)/1000) == 0):
                logging.debug(f"Train Step {i + 1}/{len(train_loader)}, Loss: {loss.data.item()}")
                #print(f"Train Step {i + 1}/{len(train_loader)}, Loss: {loss.data.item()}")

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
                if((i+1) % (len(val_loader)/1000) == 0):
                    logging.debug(f"Validation Step {i + 1}/{len(val_loader)}, Loss: {loss.data.item()}")
                #print(f"Validation Step {i + 1}/{len(val_loader)}, Loss: {loss.data.item()}")

        #print(f"Done with validation after epoch ->{epoch}")
        train_losses.append(np.mean(train_losses_epoch))
        val_losses.append(np.mean(val_losses_epoch))
        logging.info(f"--- EPOCH {epoch} AVG TRAIN LOSS: {np.mean(train_losses_epoch)}")
        logging.info(f"--- EPOCH {epoch} AVG VAL LOSS: {np.mean(val_losses_epoch)}")
    
    logging.info(f"Out of loop after epoch ->{epoch}")
    return train_losses, val_losses

def evaluate_cnn(test_loader, model, device, pixel_size = 6):
    
    true_x, true_y, true_z = [],[],[]
    mean_x, mean_y = [],[]
    sigma_x, sigma_y = [],[]
    predicted_x, predicted_y, predicted_z = [],[],[]
    with torch.no_grad():

        model.eval()
        for i, (images, positions) in enumerate(test_loader):

            images = images.to(device)
            outputs = model(images).cpu()

            for x in positions[:,0]: true_x.append(x)
            for y in positions[:,1]: true_y.append(y)
            for z in positions[:,2]: true_z.append(z)

            for x in outputs[:,0]: predicted_x.append(x)
            for y in outputs[:,1]: predicted_y.append(y)
            for z in outputs[:,2]: predicted_z.append(z)

            for img in images.cpu().squeeze().numpy():
                mu_x, mu_y, sd_x, sd_y = weighted_mean_and_sigma(img)
                mean_x.append(mu_x); mean_y.append(mu_y)
                sigma_x.append(sd_x); sigma_y.append(sd_y)

    # Convert to numpy arrays
    true_x = np.array(true_x); true_y = np.array(true_y); true_z = np.array(true_z)
    predicted_x = np.array(predicted_x); predicted_y = np.array(predicted_y); predicted_z = np.array(predicted_z)
    mean_x = np.array(mean_x); mean_y = np.array(mean_y)
    sigma_x = np.array(sigma_x); sigma_y = np.array(sigma_y)

    # Compute deltas for the NN.
    delta_x_NN = true_x - predicted_x
    delta_y_NN = true_y - predicted_y
    delta_z_NN = true_z - predicted_z

    # Compute deltas for the classical method
    delta_x_classical = true_x - pixel_size*mean_x
    delta_y_classical = true_y - pixel_size*mean_y

    tdeltas = namedtuple('tdeltas',
           'delta_x_NN, delta_y_NN, delta_z_NN, delta_x_classical, delta_y_classical')
    return tdeltas(delta_x_NN, delta_y_NN, delta_z_NN, delta_x_classical, delta_y_classical)



    