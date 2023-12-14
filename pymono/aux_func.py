import os
import logging
import numpy as np
import pandas as pd

from typing import Tuple
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def select_image_files(data_path: str, file_id: str)->Tuple[str,str]:
    """
    Return the names of an .npy file storing image data and a .csv file storing metadata
    
    """
    img_name = os.path.join(data_path, "images_" + str(file_id) + ".npy")
    lbl_name = os.path.join(data_path, "metadata_" + str(file_id) + ".csv")
    logging.debug(f"image file selected = {img_name}")
    logging.debug(f"metadata file selected = {lbl_name}")
    return img_name, lbl_name


def select_image_and_metadata(data_path: str, file_id: str)->Tuple[np.ndarray, pd.DataFrame]:
    """
    Returns a numpy vector containing image data and a PD DataFrame with metadata
    
    """
    img_name, lbl_name = select_image_files(data_path, file_id)
    mdata = pd.read_csv(lbl_name)
    imgs  = np.load(img_name)
    return imgs, mdata


def energy(data_path: str, file_id: str)->np.ndarray:
    """
    Compute the energy of the selected images by adding the contents (number of photons)
    in each pixel
    
    """
    imgs, mdata = select_image_and_metadata(data_path, file_id)
    energies = [imgs[i].sum() for i in range(0,imgs.shape[0])]
    return np.array(energies)


def mean_rms(energies: np.ndarray)->Tuple[float, float, float]:
    """
    Compute the mean, std and std/mean (FWHM) of the energy vector stored in ```energies```

    """
    return np.mean(energies), np.std(energies), 2.3*np.std(energies)/np.mean(energies)


def weighted_mean_and_sigma(image):

    # Total intensity of the image
    total_intensity = np.sum(image)

    # Indices for x and y (make (0,0) the center of the 8x8 grid)
    y_indices, x_indices = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    y_indices = np.array(y_indices) - 3.5
    x_indices = np.array(x_indices) - 3.5

    # Weighted means
    weighted_mean_x = np.sum(x_indices * image) / total_intensity
    weighted_mean_y = np.sum(y_indices * image) / total_intensity

    # Weighted standard deviations
    weighted_sigma_x = np.sqrt(np.sum(image * (x_indices - weighted_mean_x)**2) / total_intensity)
    weighted_sigma_y = np.sqrt(np.sum(image * (y_indices - weighted_mean_y)**2) / total_intensity)

    return weighted_mean_x, weighted_mean_y, weighted_sigma_x, weighted_sigma_y