import numpy as np 
import pandas as pd 
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import v2

import os

from pymono.cnn_aux import get_image_file_data
from pymono.cnn_aux import get_file_names_format1
from pymono.cnn_aux import get_img_file_metadata

class XDataset(Dataset):
    """
    Loads images to pytorch for classification as 1c (1), 2c(2), >2c(3) 
    self.dataset ->[d1, d2...] where di ->[img (normalized), scalar (1,2,3)] 
    
    """

    def __init__(self, dir_root: str, frst_file: int, lst_file: int, 
                 type=None, norm=False, mean=None, std=None, numcls=3, nc=True, random_seed=42):

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
        d2c = os.path.join(dir_root,"df2c")  # directory with images 2c

        if nc:
            dnc = os.path.join(dir_root,"dfnc")  # directory with images nc

        print(f"Running XDataset with norm = {self.norm}")
        print(f"directory for 1c = {d1c}, 2c = {d2c}")

        if nc:
            print(f"directory for  >2c = {dnc}")

        if norm == True:
            self.transf = v2.Normalize(mean=[mean], std=[std])

        if type == None or type =="1c":
            img_names, _ = get_file_names_format1(d1c)  # get images of 1c
            append_images(img_names,0)
        
        if type == None or type =="2c":
            img_names, _ = get_file_names_format1(d2c)  # get images of 2c
            append_images(img_names,1)
        
        if nc:
            if type == None or type =="nc":
                img_names, _ = get_file_names_format1(dnc)  # get images of 2c
                if numcls == 3:
                    append_images(img_names,2)
                else:
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


class RDataset(Dataset):
    """
    Loads the data to pytorch 
    self.dataset ->[d1, d2...] where di ->[img (normalized), vector (x,y,z)]
    
    """

    def __init__(self, dir_root: str, frst_file: int, lst_file: int, type=None,
                 norm=False, mean=None, std=None, random_seed=42, prnt=1000):

        def append_data(img_names, csv_name):
            test = True
            df = pd.read_csv(csv_name[0])
            if test:
                print(df.head(10))
            for i in range(frst_file, lst_file):
                images,_, img_name, imfn = get_image_file_data(img_names,i)
                
                if i%prnt == 0: 
                    print(f"image name = {img_name}")
                    print(f"image number = {imfn}")
                    print(f"number of images in file = {len(images)}")

                metadata = get_img_file_metadata(df, imfn)  # This are the events corresponding to the images
                if i%prnt == 0:
                    print(f"number of labels in file = {len(metadata)}")
            
                for img, meta in zip(images, metadata.values):
                    if test:
                        print(f"meta =>{meta}")
                        print(f"meta =>{meta[2:5]}")
                        test = False
                    
                    self.dataset.append((img, meta[2:5]))
                    
        self.norm=norm 
        self.dataset = []
        self.transf = None 

        print(f"Running rDataset with norm = {self.norm}")
        
        if norm == True:
            self.transf = v2.Normalize(mean=[mean], std=[std])

        d1c = os.path.join(dir_root,"df1c")  # directory with images 1c
        d2c = os.path.join(dir_root,"df2c")  # directory with images 2c
        #dnc = os.path.join(dir_root,"dfnc")  # directory with images nc

        if type == None or type =="1c":
            print(f"Loading files in directory d1c with indexes: {frst_file}, {lst_file}")
            
            img_names, csv_name = get_file_names_format1(d1c)
            append_data(img_names, csv_name)
                    
        if type == None or type =="2c":
            print(f"Loading files in directory d2c with indexes: {frst_file}, {lst_file}")
            
            img_names, csv_name = get_file_names_format1(d2c)
            append_data(img_names, csv_name)

        self.si = list(range(len(self.dataset)))
        print(f"Before shufle: length si: {len(self.si)}, si->{self.si[0:10]}")
        np.random.seed(random_seed)
        np.random.shuffle(self.si)
        print(f"After shufle: length si: {len(self.si)}, si->{self.si[0:10]}")
         
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        #image, position = self.dataset[self.si[idx]]
        image, position = self.dataset[idx]
        image = torch.tensor(image, dtype=torch.float).unsqueeze(0) # Add channel dimension
        position = torch.tensor(position, dtype=torch.float)

        if self.norm == True:
            image = self.transf(image)

        return image, position


class R2Dataset(Dataset):
    """
    Loads the data to pytorch 
    self.dataset ->[d1, d2...] where di ->[img (normalized), vector (x,y,z)]
    
    """

    def __init__(self, dir_root: str, frst_file: int, lst_file: int, 
                 norm=False, mean=None, std=None, random_seed=42, bcnt=False, type=None, prnt=1000):

        def append_data(img_names, csv_name):
            test = True
            df = pd.read_csv(csv_name[0])
            if test:
                print(df.head(10))
            for i in range(frst_file, lst_file):
                images,_, img_name, imfn = get_image_file_data(img_names,i)
                
                if i%prnt == 0: 
                    print(f"image name = {img_name}")
                    print(f"image number = {imfn}")
                    print(f"number of images in file = {len(images)}")

                metadata = get_img_file_metadata(df, imfn)  # This are the events corresponding to the images
                if i%prnt == 0:
                    print(f"number of labels in file = {len(metadata)}")
            
                for img, meta in zip(images, metadata.values):
                    
                    if bcnt:
                        e1 = meta[1]
                        x1 = meta[2]
                        y1 = meta[3]
                        z1 = meta[4]
                        e2 = meta[6]
                        x2 = meta[7]
                        y2 = meta[8]
                        z2 = meta[9]
                        et = meta[11]
                        xb = (x1*e1 + x2 * e2)/et
                        yb = (y1*e1 + y2 * e2)/et
                        zb = (z1*e1 + z2 * e2)/et
                        if test:
                            print(f"x1 =>{x1}, y1 =>{y1}, z1 =>{z1}")
                            print(f"x2 =>{x2}, y2 =>{y2}, z2 =>{z2}")
                            print(f"xb =>{xb}, yb =>{yb}, zb =>{zb}")
                            test = False
                        self.dataset.append((img, (xb,yb,zb)))
                    else:
                        self.dataset.append((img, np.concatenate((meta[2:5], meta[7:10]))))
                    
        self.norm=norm 
        self.dataset = []
        self.transf = None 

        print(f"Running rDataset with norm = {self.norm}")
        
        if norm == True:
            self.transf = v2.Normalize(mean=[mean], std=[std])

        d1c = os.path.join(dir_root,"df1c")  # directory with images 1c
        d2c = os.path.join(dir_root,"df2c")  # directory with images 2c
        #dnc = os.path.join(dir_root,"dfnc")  # directory with images nc

        if type == None or type =="1c":
            print(f"Loading files in directory d1c with indexes: {frst_file}, {lst_file}")
            
            img_names, csv_name = get_file_names_format1(d1c)
            append_data(img_names, csv_name)
                    
        if type == None or type =="2c":
            print(f"Loading files in directory d2c with indexes: {frst_file}, {lst_file}")
            
            img_names, csv_name = get_file_names_format1(d2c)
            append_data(img_names, csv_name)


        self.si = list(range(len(self.dataset)))
        print(f"Before shufle: length si: {len(self.si)}, si->{self.si[0:10]}")
        np.random.seed(random_seed)
        np.random.shuffle(self.si)
        print(f"After shufle: length si: {len(self.si)}, si->{self.si[0:10]}")
         
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        #image, position = self.dataset[self.si[idx]]
        image, position = self.dataset[idx]
        image = torch.tensor(image, dtype=torch.float).unsqueeze(0) # Add channel dimension
        position = torch.tensor(position, dtype=torch.float)

        if self.norm == True:
            image = self.transf(image)

        return image, position
    
