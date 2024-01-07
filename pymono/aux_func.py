import os
import logging
import numpy as np
import pandas as pd

from typing import Tuple
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def select_image_files(data_path: str, file_id: str, pad=False)->Tuple[str,str]:
    """
    Return the names of an .npy file storing image data and a .csv file storing metadata
    
    """
    if pad:
        img_name = os.path.join(data_path, "images_0" + str(file_id) + ".npy")
    else:
        img_name = os.path.join(data_path, "images_" + str(file_id) + ".npy")

    if pad:
        lbl_name = os.path.join(data_path, "metadata_0" + str(file_id) + ".csv")
    else:
        lbl_name = os.path.join(data_path, "metadata_" + str(file_id) + ".csv")
    logging.debug(f"image file selected = {img_name}")
    logging.debug(f"metadata file selected = {lbl_name}")
    return img_name, lbl_name


def select_image_and_metadata(data_path: str, file_id: str, pad=False)->Tuple[np.ndarray, pd.DataFrame]:
    """
    Returns a numpy vector containing image data and a PD DataFrame with metadata
    
    """
    img_name, lbl_name = select_image_files(data_path, file_id, pad)
    mdata = pd.read_csv(lbl_name)
    imgs  = np.load(img_name)
    return imgs, mdata


def energy(data_path: str, file_id: str, pad=False)->np.ndarray:
    """
    Compute the energy of the selected images by adding the contents (number of photons)
    in each pixel
    
    """
    imgs, mdata = select_image_and_metadata(data_path, file_id, pad)
    energies = [imgs[i].sum() for i in range(0,imgs.shape[0])]
    return np.array(energies)


def corrected_energy(data_path: str, file_id: str, 
                     energy_map: str, energy_bins: str, pad=False)->np.ndarray:
    """
    Compute the energy of the selected images by adding the contents (number of photons)
    in each pixel, then correct it using energy map 
    
    """
    def corr_factor(i,energies, positions, h3d, binsx, binsy, binsz):
        pos = positions[i]
        ix = min(np.digitize(pos[0], binsx) -1, h3d.shape[0]-1)
        iy = min(np.digitize(pos[1], binsy) -1, h3d.shape[1]-1)
        iz = min(np.digitize(pos[2], binsz) -1, h3d.shape[2]-1)
        
        cf = h3d[ix, iy, iz]
        ce = energies[i]/cf
        return ce 

    imgs, mdata = select_image_and_metadata(data_path, file_id, pad)
    positions   = [[mdata.iloc[i].initial_x, mdata.iloc[i].initial_y,
                    mdata.iloc[i].initial_z] for i in range(0,mdata.shape[0])]
    
    h3d = np.load(energy_map)
    binsx = np.load("x_"+energy_bins)
    binsy = np.load("y_"+energy_bins)
    binsz = np.load("z_"+energy_bins)

    energies = [imgs[i].sum() for i in range(0,imgs.shape[0])]
    cene = [corr_factor(i,energies, positions, 
                        h3d, binsx, binsy, binsz) for i in range(0,len(energies))]

    return np.array(cene)


def mean_rms(energies: np.ndarray, fwhm_only=False)->Tuple[float, float, float]:
    """
    Compute the mean, std and std/mean (FWHM) of the energy vector stored in ```energies```

    """
    if fwhm_only:
        return 2.3*np.std(energies)/np.mean(energies)
    else:
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


def energy_cube(data_path, file_number=0, bins = (10, 10, 10), pad=False):
    """
    For the data in ```file_number``, compute a numpy histogramdd (in 3D) in which every axis 
    is the position of the true interaction and the weight is the energy recorded by the SiPMs,
    that is the sume of the pixels of the image. 

    """
    imgs, mdata = select_image_and_metadata(data_path, file_number, pad) 
    positions = np.array([[mdata.iloc[i].initial_x, mdata.iloc[i].initial_y,
                           mdata.iloc[i].initial_z] for i in range(0,mdata.shape[0])])
    energies = np.array([np.sum(imgs[i]) for i in range(0, mdata.shape[0])])
    h3e      = np.histogramdd(positions, bins = bins, density=False, weights=energies)
    return h3e


def energy_h3d(data_path, file_range=(0,99), bins = (10, 10, 10), compute=True, 
               file_name_h3d="h3d.npy", file_name_h3e="h3e.npy", pad=False):
    """
    For the data in ```file_range``, compute a numpy histogramdd (in 3D) in which every axis 
    is the position of the true interaction and the weight is the energy recorded by the SiPMs,
    that is the sume of the pixels of the image. The histogram is the mean of the individual
    histograms obtained for each file

    """
    if compute:
        h3ds = [energy_cube(data_path, i, bins, pad) for i in range(*file_range)]
        h3d  = np.mean([h3ds[i][0] for i in range(*file_range)], axis=0)
        h3e  = h3ds[0][1]
        emax = h3d.max()
        for i in range(0,h3d.shape[2]): 
            #h3d[:,:,i] = h3d[:,:,i]/np.amax(h3d[:,:,i])
            h3d[:,:,i] = h3d[:,:,i]/emax
        np.save(file_name_h3d, h3d)
        np.save("x_"+file_name_h3e, h3e[0])
        np.save("y_"+file_name_h3e, h3e[1])
        np.save("z_"+file_name_h3e, h3e[2])
    else:
        h3d = np.load(file_name_h3d)
        h3ex = np.load("x_"+file_name_h3e)
        h3ey = np.load("y_"+file_name_h3e)
        h3ez = np.load("z_"+file_name_h3e)
        h3e=[h3ex, h3ey, h3ez]

    return h3d, h3e