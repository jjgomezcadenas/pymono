import torch 
import torch.nn as nn
import numpy as np 
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch.nn.functional as F
import sys

from collections import namedtuple

from cnn_aux import files_list_npy_csv

class GIDataset(Dataset):
    """
    Loads the data to pytorch 
    self.dataset ->[d1, d2...] where di ->[img (normalized), vector (x,y,z)]
    
    - Variable xyze intoduced to allow two types of format:
    1. csv files with format xyz (set xyze == False)
    2. csv files with format xyze (set xyze == True)
   
    """

    def __init__(self, data_path: str, frst_file: int, lst_file: int, 
                 norm=False, resize=False, mean=None, std=None, size=None,
                 twoc=0): 
        """
        twoc = 0 selects one cluster
        twoc = 1 selects two clusters
        twoc = 2 selects two clusters with energy
        """

        self.norm=norm 
        self.resize=resize
        self.dataset = []
        self.transf = None 

        print(f"Running GIDataset with norm = {self.norm}, resize={self.resize}")
        
        if norm == True and resize == True:
            print("defining transform: componse Resize and Normalize")
            self.transf = v2.Compose([
                            v2.Resize(size=size, antialias=True),
                            v2.Normalize(mean=[mean], std=[std])])
                    
        elif norm == True:
            self.transf = v2.Normalize(mean=[mean], std=[std])
                    
        elif resize == True:
            self.transf = v2.Resize(size=size, antialias=True)
                   

        img_name, lbl_name, indx = files_list_npy_csv(data_path)
        print(f"Loading files with indexes: {indx[frst_file:lst_file]}")

        for i in indx[frst_file:lst_file]:
            print(f"Loading {img_name}_{i}.npy, {lbl_name}_{i}.csv")
            
            images = np.load(f'{data_path}/{img_name}_{i}.npy')
            metadata = pd.read_csv(f'{data_path}/{lbl_name}_{i}.csv').drop(["event", "etot", "ntrk", "t1", "t2"], axis=1)
            metadata = metadata[['x1', 'y1', 'z1','x2','y2','z2','e1','e2']]
            if twoc == 1:
                metadata = metadata.drop(["e1", "e2"], axis=1)
           
            for img, meta in zip(images, metadata.values):
                    if twoc>0:
                        self.dataset.append(((img, meta)))
                    else:
                        self.dataset.append(((img, meta[0:3])))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, meta = self.dataset[idx]
        image = torch.tensor(image, dtype=torch.float).unsqueeze(0) # Add channel dimension
        meta = torch.tensor(meta, dtype=torch.float)

        if self.norm == True or self.resize == True:
            image = self.transf(image)

        return image, meta




class CNN_basic(nn.Module):
    """
    Defines a convolutional network with a basic architecture:
    convolution (3x3) , reLU batch norm and MaxPool: (8,8,1) => (4,4,128)
    convolution (2x2) , reLU batch norm and MaxPool: (4,4,128) => (2,2,256)
    convolution (2x2) , reLU batch norm and MaxPool: (2,2,256) => (1,1,512)
    drop (optional) 
    linear layer 512 => 

    twoc = 0 selects one cluster          512 => 3
        twoc = 1 selects two clusters     512 => 6
        twoc = 2 selects two clusters with energy  512 => 8

    Input to the network are the pixels of the pictures, output (x,y,z)

    """
    def __init__(self, chi=128, dropout=False, dropout_fraction=0.2, twoc = 0):
        super().__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(1, chi, 3, padding=1) 
        self.bn1   = nn.BatchNorm2d(chi)
        self.conv2 = nn.Conv2d(chi, chi*2, 2, padding=1)
        self.bn2   = nn.BatchNorm2d(chi*2)
        self.conv3 = nn.Conv2d(chi*2, chi*4, 2, padding=1)
        self.bn3   = nn.BatchNorm2d(chi*4)
        self.pool = nn.MaxPool2d(2, 2)

        if twoc == 2:
            self.fc0 = nn.Linear(chi*4, 8)
        elif twoc == 1:
            self.fc0 = nn.Linear(chi*4, 6)
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


def evaluate_cnn(test_loader, model, device, twoc=0):
    """
    valuate the CNN returning the difference between true and predicted for the three coordinates

    """
    true_x, true_y, true_z = [],[],[]
    if twoc > 0:
        true_x2, true_y2, true_z2 = [],[],[]
    if twoc == 2:
        true_e = []
        true_e2 = []

    predicted_x, predicted_y, predicted_z = [],[],[]
    if twoc > 0:
        predicted_x2,  predicted_y2,  predicted_z2 = [],[],[]
    if twoc == 2:
        predicted_e = []
        predicted_e2 = []
         
    with torch.no_grad():

        model.eval()
        for i, (images, positions) in enumerate(test_loader):

            images = images.to(device)
            outputs = model(images).cpu()

            for x in positions[:,0]: true_x.append(x)
            for y in positions[:,1]: true_y.append(y)
            for z in positions[:,2]: true_z.append(z)
            if twoc > 0:
                for x in positions[:,3]: true_x2.append(x)
                for y in positions[:,4]: true_y2.append(y)
                for z in positions[:,5]: true_z2.append(z)
            
            if twoc == 2:
                for e in positions[:,6]: true_e.append(e)
                for e in positions[:,7]: true_e2.append(e)

            for x in outputs[:,0]: predicted_x.append(x)
            for y in outputs[:,1]: predicted_y.append(y)
            for z in outputs[:,2]: predicted_z.append(z)
                
            if twoc > 0:
                for x in outputs[:,3]: predicted_x2.append(x)
                for y in outputs[:,4]: predicted_y2.append(y)
                for z in outputs[:,5]: predicted_z2.append(z)
            if twoc == 2:
                for e in outputs[:,6]: predicted_e.append(e)
                for e in outputs[:,7]: predicted_e2.append(e)

    # Convert to numpy arrays
    true_x = np.array(true_x); true_y = np.array(true_y); true_z = np.array(true_z)
    if twoc >0:
        true_x2 = np.array(true_x2); true_y2 = np.array(true_y2); true_z2 = np.array(true_z2)

    if twoc == 2:
        true_e = np.array(true_e)
        true_e2 = np.array(true_e2)

    predicted_x = np.array(predicted_x) 
    predicted_y = np.array(predicted_y); predicted_z = np.array(predicted_z)
    if twoc > 0:
       predicted_x2 = np.array(predicted_x2) 
       predicted_y2 = np.array(predicted_y2); predicted_z2 = np.array(predicted_z2)
    if twoc == 2:
       predicted_e = np.array(predicted_e)
       predicted_e2 = np.array(predicted_e2)

    # Compute deltas for the NN.
    delta_x_NN = true_x - predicted_x
    delta_y_NN = true_y - predicted_y
    delta_z_NN = true_z - predicted_z

    if twoc>0:
        delta_x_NN2 = true_x2 - predicted_x2
        delta_y_NN2 = true_y2 - predicted_y2
        delta_z_NN2 = true_z2 - predicted_z2
    if twoc == 2:
        delta_e_NN = true_e - predicted_e
        delta_e_NN2 = true_e2 - predicted_e2

    if twoc == 2:
        tdeltas = namedtuple('tdeltas',
            'delta_x_NN, delta_y_NN, delta_z_NN, delta_e_NN, delta_x_NN2, delta_y_NN2, delta_z_NN2, delta_e_NN2')
        return tdeltas(delta_x_NN, delta_y_NN, delta_z_NN, delta_e_NN, delta_x_NN2, delta_y_NN2, delta_z_NN2, delta_e_NN2)
    elif twoc == 1:
        tdeltas = namedtuple('tdeltas',
            'delta_x_NN, delta_y_NN, delta_z_NN, delta_x_NN2, delta_y_NN2, delta_z_NN2')
        return tdeltas(delta_x_NN, delta_y_NN, delta_z_NN, delta_x_NN2, delta_y_NN2, delta_z_NN2)
    else:
        tdeltas = namedtuple('tdeltas',
            'delta_x_NN, delta_y_NN, delta_z_NN')
        return tdeltas(delta_x_NN, delta_y_NN, delta_z_NN)