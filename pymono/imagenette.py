import numpy as np 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import sys 
import os

from pymono.config import imagenette_dir
import logging

logging.basicConfig(stream=sys.stdout,
                    format='%(levelname)s:%(message)s', level=logging.DEBUG)




def imagenette_data_loader(batch_size=32, shuffle=True, norm = True, test=False):
    
    print("Reading imaginette")
    
    TRAIN_DATA_DIR = os.path.join(imagenette_dir, "train")
    TEST_DATA_DIR = os.path.join(imagenette_dir, "val")
  
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # define transforms
    stransform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
    ])
    #transform = transforms.Compose([
    #        transforms.Resize((224,224)),
    #        transforms.ToTensor(),
    #        normalize,
    #])

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])


    if test:
        if norm:
            print("case test and norm")
            dataset = datasets.ImageFolder(TRAIN_DATA_DIR, transform=transform)
        else:
            print("case test and norm false")
            dataset = datasets.ImageFolder(TRAIN_DATA_DIR, transform=stransform)

        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


    # Load ImageNette dataset
    trainset = datasets.ImageFolder(TRAIN_DATA_DIR, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)

    testset= datasets.ImageFolder(TEST_DATA_DIR, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return (trainloader, testloader)

