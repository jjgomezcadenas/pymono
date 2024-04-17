import numpy as np 

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import v2

import os

from pymono.cnn_aux import get_image_file_data
from pymono.cnn_aux import get_file_names_format1

class XDataset(Dataset):
    """
    Loads images to pytorch for classification as 1c (1), 2c(2), >2c(3) 
    self.dataset ->[d1, d2...] where di ->[img (normalized), scalar (1,2,3)] 
    
    """

    def __init__(self, dir_root: str, frst_file: int, lst_file: int, 
                 norm=False, mean=None, std=None, random_seed=42):

        def append_images(img_names, imgt_type):
            print(f" Image type = {imgt_type}")
            
            for img_file_index in range(frst_file, lst_file):
                images, _, _, _ = get_image_file_data(img_names,img_file_index)
                #print(f"image file path = {imgf_path}")
                #print(f"number of images in file = {len(images)}")
                
                for img in images:
                    #print(f"append image =>{img}")
                    #print(f"append img_tupe =>{imgt_type}")
                    self.dataset.append((img, imgt_type))
                    
        self.norm=norm 
        self.dataset = []
        self.transf = None 

        d1c = os.path.join(dir_root,"df1c")  # directory with images 1c
        d2c = os.path.join(dir_root,"df2c")  # directory with images 1c
        dnc = os.path.join(dir_root,"dfnc")  # directory with images nc

        print(f"Running XDataset with norm = {self.norm}")
        print(f"directory for 1c = {d1c}, 2c = {d2c}, >2c = {dnc}")

        if norm == True:
            self.transf = v2.Normalize(mean=[mean], std=[std])

        img_names, _ = get_file_names_format1(d1c)  # get images of 1c
        append_images(img_names,0)
        
        img_names, _ = get_file_names_format1(d2c)  # get images of 2c
        append_images(img_names,1)

        img_names, _ = get_file_names_format1(dnc)  # get images of 2c
        append_images(img_names,1)

        self.si = list(range(len(self.dataset)))
        print(f"Before shufle: length si: {len(self.si)}, si->{self.si[0:10]}")
        np.random.seed(random_seed)
        np.random.shuffle(self.si)
        print(f"After shufle: length si: {len(self.si)}, si->{self.si[0:10]}")

        
    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, idx):
        image, imgtype = self.dataset[self.si[idx]]
        image = torch.tensor(image, dtype=torch.float).unsqueeze(0) # Add channel dimension
        imgtype = torch.tensor(imgtype)

        if self.norm == True:
            image = self.transf(image)

        return image, imgtype

