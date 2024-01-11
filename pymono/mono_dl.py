import numpy as np 
import pandas as pd
from collections import namedtuple
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import v2
import glob 
import os

def files_list_npy_csv(data_path):
    """
    Return the list of .npy file storing image data and of .csv file storing metadata
    in the directory data_path
    
    """
    def select_files(ext="*.npy"):
        npys = os.path.join(data_path, ext)
        return glob.glob(npys)

    images = [f1.split("/")[-1] for f1 in select_files(ext="*.npy")]
    metadata = [f1.split("/")[-1] for f1 in select_files(ext="*.csv")]
    imn = [int(im.split(".")[0].split("_")[1]) for im in images]
    mdn = [int(im.split(".")[0].split("_")[1]) for im in metadata]
    assert (np.sort(mdn) == np.sort(imn)).all()

    imx=images[0].split(".")[0].split("_")[0]
    mdx = metadata[0].split(".")[0].split("_")[0]
    return imx, mdx, np.sort(imn)


class MonoDataset(Dataset):
    """
    Loads the data to pytorch 
    self.dataset ->[d1, d2...] where di ->[img (normalized), vector (x,y,z)]
    """

    def __init__(self, data_path: str, frst_file: int, lst_file: int, 
                 norm=False, resize=False, mean=None, std=None, size=None):

        self.norm=norm 
        self.resize=resize
        self.dataset = []
        self.transf = None 

        print(f"Running Monoataset with norm = {self.norm}, resize={self.resize}")
        
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

        self.verbose = True
        for i in indx[frst_file:lst_file]:
            images = np.load(f'{data_path}/{img_name}_{i}.npy')
            metadata = pd.read_csv(f'{data_path}/{lbl_name}_{i}.csv')
            
            for img, meta in zip(images, metadata.values):
                self.dataset.append(((img, meta[1:])))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, position = self.dataset[idx]
        image = torch.tensor(image, dtype=torch.float).unsqueeze(0) # Add channel dimension
        position = torch.tensor(position, dtype=torch.float)

        if self.norm == True or self.resize == True:
            image = self.transf(image)

        return image, position
    

def mono_data_loader(dataset, 
                     batch_size=1000, 
                     train_fraction=0.7, 
                     val_fraction=0.2):
    
    def split_data(dataset, train_fraction, val_fraction):
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

    
    def load_and_split(dataset, batch_size, train_fraction, val_fraction):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"Loaded {len(dataset)} events")
        
        # Split the data into training, validation, and test sets
        trsz = split_data(dataset, train_fraction, val_fraction)
        print(f" train size = {trsz.train_size}")
        print(f" val size = {trsz.val_size}")
        print(f" test size = {trsz.test_size}")
        print(f" train indices = {trsz.train_indices}")
        print(f" val indices = {trsz.val_indices}")
        print(f" test indices = {trsz.test_indices}")
        return data_loader, trsz
    
    def get_subsets(dataset, trsz):
        train_dataset = torch.utils.data.Subset(dataset, trsz.train_indices)
        val_dataset = torch.utils.data.Subset(dataset, trsz.val_indices)
        test_dataset = torch.utils.data.Subset(dataset, trsz.test_indices)

        print(f"{len(train_dataset)} training events ({100*len(train_dataset)/len(dataset)}%)")
        print(f"{len(val_dataset)} validation events ({100*len(val_dataset)/len(dataset)}%)")
        print(f"{len(test_dataset)} test events ({100*len(test_dataset)/len(dataset)}%)")
        return train_dataset, val_dataset, test_dataset 

    def get_loaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size):
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            return train_loader, val_loader, test_loader

    
    data_loader, trsz = load_and_split(dataset, 
                                       batch_size, 
                                       train_fraction, 
                                       val_fraction)
    
    train_dataset, val_dataset, test_dataset = get_subsets(dataset, trsz)

    train_loader, val_loader, test_loader = get_loaders(train_dataset, 
                                                        val_dataset, 
                                                        test_dataset, 
                                                        batch_size)
    
    return data_loader, train_loader, val_loader, test_loader


