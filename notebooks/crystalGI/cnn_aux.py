import numpy as np
import pandas as pd
from collections import namedtuple
import os
import glob
from pathlib import Path


import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import v2


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


def select_image_and_lbl(data_path, file_id):
    """
    Returns a numpy vector containing image data and a PD DataFrame with metadata
    
    """
    img_name, lbl_name, indx = files_list_npy_csv(data_path)
    
    if file_id < 0 or file_id > len(indx) -1:
        assert False
    else:
        img_fname = f"{img_name}_{indx[file_id]}.npy"
        lbl_fname = f"{lbl_name}_{indx[file_id]}.csv"
        print(f"Selected files: img = {img_fname}, metdata = {lbl_fname}")
        imgs  = np.load(os.path.join(data_path,img_fname))
        mdata = pd.read_csv(os.path.join(data_path, lbl_fname))
        return imgs, mdata




def get_gamma_position_in_pixels(df, x_spatial, y_spatial, debug=False):
    """
    Return the gamma position in the pixel coordinate system
    df is a series containing the event, x1, y1, z1, e1, x2... 
    
    """

    def transform_coordinates(x, y, x_spatial, y_spatial, x_min2=0,   x_max2=8,  y_min2=0,   y_max2 = 8):
        """
        Transform coordinates from space RS to pixel RS
        """
                        
        x_min1=x_spatial[0] 
        x_max1=x_spatial[-1] 
        y_min1=y_spatial[0]
        y_max1 = y_spatial[-1]
    
        if x < x_min1:
            x = x_min1
    
        if x > x_max1:
            x = x_max1
    
        if y < y_min1:
            y = y_min1
    
        if y > y_max1:
            y = y_max1
     
        # Apply the transformation for x and y
        x_new = ((x - x_min1) / (x_max1 - x_min1)) * (x_max2 - x_min2) + x_min2
        y_new = ((y - y_min1) / (y_max1 - y_min1)) * (y_max2 - y_min2) + y_min2
    
        return x_new, y_new
    
    xt1, yt1 = transform_coordinates(df.x1, df.y1, 
                                     x_spatial, y_spatial)

    xt2, yt2 = transform_coordinates(df.x2, df.y2, 
                                     x_spatial, y_spatial)
    

    if debug:
        print(f"xg1 = {df.x1:.2f}, yg1 ={df.y1:.2f}")
        print(f"xt1 = {xt1:.2f}, yt1 ={yt1:.2f}")
        print(f"xg2 = {df.x2:.2f}, yg2 ={df.y2:.2f}")
        print(f"xt2 = {xt2:.2f}, yt2 ={yt2:.2f}")

    return xt1, yt1,xt2, yt2


def get_energy(imgs):
    """
    Compute the energy of the images (imgs) by adding the contents (number of photons)
    in each pixel
    
    """
    energies = [imgs[i].sum() for i in range(0,imgs.shape[0])]
    return np.array(energies)


def mean_rms(energies, fwhm_only=False):
    """
    Compute the mean, std and std/mean (FWHM) of the energy vector stored in ```energies```

    """
    if fwhm_only:
        return 2.3*np.std(energies)/np.mean(energies)
    else:
        return np.mean(energies), np.std(energies)/np.mean(energies), 2.3*np.std(energies)/np.mean(energies)





