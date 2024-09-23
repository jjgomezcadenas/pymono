import numpy as np
import pandas as pd
from collections import namedtuple
import os
import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from cnn_aux import files_list_npy_csv, select_image_and_lbl, get_energy, mean_rms
from cnn_models import GIDataset, CNN_basic, evaluate_cnn
from cnn_plot import histoplot, plot_images,scatter_xyz, scatter_xyze

from pymono.mono_dl import mono_data_loader
from pymono.plt_funcs import plot_images_ds, plot_loss, plotxyz
from pymono.cnn_eval import train_cnn, evaluate_cnn
from pymono.cnn_fit import fit_tdeltas, plotfxyz
from cnn_plot import plotxyz_twoc

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def main():
    start_time = time.time() 
    data_dir = Path(os.environ['DATA'])
    imgdir = os.path.join(data_dir,"G4Prods/crystalGI", "BGOH1")
    
    ## CNN run 1

    first_file = 0  # initial file indx
    last_file  = 100  # lasta file indx
    batch_size = 1000  # Batch size
    train_fraction=0.7 
    val_fraction=0.2

    dataset = GIDataset(imgdir, first_file, last_file,twoc=1) 
    data_loader, train_loader, val_loader, test_loader = mono_data_loader(dataset, 
                                                                      batch_size=batch_size, 
                                                                      train_fraction=train_fraction, 
                                                                      val_fraction=val_fraction)

    for images, positions in train_loader:
        print(images.size())
        print(positions.size())
        print(images[0,0,:,:])
    break

    model = CNN_basic(chi=128, dropout=True, dropout_fraction=0.2, twoc=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_losses, val_losses = train_cnn(train_loader, val_loader, model, optimizer, device, criterion, 
                                     batch_size, epochs=10, iprnt=100)
    
    end_time = time.time()  # Stop the clock
    execution_time = end_time - start_time
    print(f"Finished: Execution time: {execution_time:.2f} seconds")

# Main program entry point
if __name__ == "__main__":
    main()
