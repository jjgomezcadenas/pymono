import numpy as np
from numpy.typing import NDArray

import pandas as pd
from collections import namedtuple
import os
import glob
from pathlib import Path

from typing import List, Tuple


def get_file_names_format1(path : str) -> Tuple[List[str], List[str]]:
    """
    Returns the names of the file in directory specified by path.
    Assumes format1 defined as:

    - images files are of the form: images_n.npy, where n is a number.
    - labels are stored in a single file (labels.csv)
    """
    fnpy = glob.glob(os.path.join(path, "*.npy"))
    fcsv = glob.glob(os.path.join(path, "*.csv"))
    images = [f1.split("/")[-1] for f1 in fnpy]
    imn = [int(im.split(".")[0].split("_")[1]) for im in images]
    imns = np.sort(imn)
    names_i = [f"{path}/images_{i}.npy" for i in imns]
    return names_i, fcsv


def get_image_file_data(img_names : List[str],img_file_index=0)->Tuple[List[NDArray], str, str, int]:
    """
    Returns the images corresponding to the file specified by img_file_index
    imgs: list of images
    img_path: complete path of image specified by img_file_index
    img_name: name of image specified by img_file_index
    imn: number of image specified by img_file_index
    """
    iimg = img_file_index
    imgs  = np.load(img_names[iimg])  # loads the images corresponding to the file specii
    img_path = img_names[iimg]
    img_name = img_path.split("/")[-1]
    imn = int(img_name.split(".")[0].split("_")[1])
    return imgs, img_path, img_name, imn


def get_img_file_metadata(df :pd.DataFrame, img_file_number : int)->pd.DataFrame:
    """
    Return the metadata corresponding to the img_file number.
    dfs is a data frame which contains the labels of the images in the file specified by img_file_number
    """
    evt_id = img_file_number * 10000
    dfs = df[df["event_id"]>=evt_id]  # Images in first file have event numbers between 10,000 and 20,000
    dfs = dfs[dfs["event_id"]<evt_id + 10000]
    return dfs


def get_energy2(imgs : List[NDArray])->List[float]:
    """
    Compute the energy of the selected images by adding the contents (number of photons)
    in each pixel

    imgs is a file containing the imaged readout in the directory
    The images selected are those contained in the file especified by img_file_number
    
    """
  
    energies = [imgs[i].sum() for i in range(0,imgs.shape[0])]
    return np.array(energies)

def get_means_stds2(dir):
    """

    Compute the means and stds of the images in dir
    
    """
    means =[]
    stds =[]
    img_names, _ = get_file_names_format1(dir)
    print(f"files in dir: {len(img_names)}")
    for i,img in enumerate(img_names):
        images = np.load(img)
        if i == 1:
            print(f"shape -> {images.shape}")
            print(f"mean img0 ={np.mean(images[0,:,:])}")
            print(f"std  img0 ={np.std(images[0,:,:])}")
        means.append(np.mean(images, axis=(1,2)))
        stds.append(np.std(images, axis=(1,2)))
    return means, stds