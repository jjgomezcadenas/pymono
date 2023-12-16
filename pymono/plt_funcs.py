import matplotlib.pyplot as plt
import numpy as np 
from pymono.aux_func import mean_rms

def plot_true_positions(mdata):
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    flat_axes = axes.ravel()
    ax0, ax1, ax2 = flat_axes[0], flat_axes[1], flat_axes[2]
    
    ax0.hist(mdata['initial_x'].to_numpy(), bins=50)
    ax0.set_xlabel("true x",fontsize=14)
    ax0.set_ylabel("Counts/bin",fontsize=14)
    
    ax1.hist(mdata['initial_y'].to_numpy(), bins=50)
    ax1.set_xlabel("true y",fontsize=14)
    ax1.set_ylabel("Counts/bin",fontsize=14)
    
    ax2.hist(mdata['initial_z'].to_numpy(), bins=50)
    ax2.set_xlabel("true z",fontsize=14)
    ax2.set_ylabel("Counts/bin",fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_images(imgs, mdata, img_numbers, pixel_size = 6, grid_size=8):
    
    if len(img_numbers) > 8:
        print("range too large, can plot a max of 8 images")
        return 0
        
    fig, axs = plt.subplots(2, 4,figsize=(14, 4))        
    ftx = axs.ravel()
    for ev, i in enumerate(img_numbers):        
        x_evt = (mdata['initial_y'][ev] + pixel_size*grid_size/2)/pixel_size - 0.5
        y_evt = (mdata['initial_x'][ev] + pixel_size*grid_size/2)/pixel_size - 0.5
        
        ftx[i].imshow(imgs[ev,:,:])
        ftx[i].plot([x_evt],[y_evt],'o',color='red')


def plot_energies(ene_light6x6, ene_light_all_6x6,  ene_light3x3, ene_dark6x6, num_bins = 50):

    mean6x6, std6x6, fwhm6x6    = mean_rms(ene_light6x6)
    mean6x6a, std6x6a, fwhm6x6a = mean_rms(ene_light_all_6x6)
    mean3x3, std3x3, fwhm3x3    = mean_rms(ene_light3x3)
    mean6x6d, std6x6d, fwhm6x6d = mean_rms(ene_dark6x6)


    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    flat_axes = axes.ravel()
    ax0, ax1, ax2, ax3 = flat_axes[0], flat_axes[1], flat_axes[2], flat_axes[3]
    
    _, _, _ = ax0.hist(ene_light6x6, num_bins, label=f"$\sigma$ (FWHM) = {fwhm6x6:.2f}")
    ax0.set_xlabel('Energy ')
    ax0.set_ylabel('Events/bin')
    ax0.set_title('Sum of energies light6x6')
    ax0.legend()

    _, _, _ = ax3.hist(ene_light_all_6x6, num_bins,label=f"$\sigma$ (FWHM) = {fwhm6x6a:.2f}")
    ax3.set_xlabel('Energy ')
    ax3.set_ylabel('Events/bin')
    ax3.set_title('Sum of energies light6x6 all reflectant')
    ax3.legend()

    _, _, _ = ax1.hist(ene_light3x3, num_bins,label=f"$\sigma$ (FWHM) = {fwhm3x3:.2f}")
    ax1.set_xlabel('Energy ')
    ax1.set_ylabel('Events/bin')
    ax1.set_title('Sum of energies light3x3')
    ax1.legend()

    _, _, _ = ax2.hist(ene_dark6x6, num_bins,label=f"$\sigma$ (FWHM) = {fwhm6x6d:.2f}")
    ax2.set_xlabel('Energy ')
    ax2.set_ylabel('Events/bin')
    ax2.set_title('Sum of energies dark6x6')
    ax2.legend()

    fig.tight_layout()
    plt.show()


def plot_true_predicted(tdeltas, nbins = 50):
    nbins = 50

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    flat_axes = axes.ravel()
    ax0, ax1 = flat_axes[0], flat_axes[1]

    ax0.hist(tdeltas.delta_x_NN, bins=nbins, 
            label=f"x ($\sigma$ = {np.std(tdeltas.delta_x_NN):.2f})", alpha=0.7)
    ax0.hist(tdeltas.delta_y_NN, bins=nbins, 
            label=f"y ($\sigma$ = {np.std(tdeltas.delta_y_NN):.2f})", alpha=0.7)
    ax0.hist(tdeltas.delta_z_NN, bins=nbins, 
            label=f"z ($\sigma$ = {np.std(tdeltas.delta_z_NN):.2f})", alpha=0.7)
    ax0.set_xlabel("NN (True - Predicted) Positions",fontsize=14)
    ax0.set_ylabel("Counts/bin",fontsize=14)
    ax0.legend()

    ax1.hist(tdeltas.delta_x_classical, bins=nbins, 
            label=f"x ($\sigma$ = {np.std(tdeltas.delta_x_classical):.2f})", alpha=0.7)
    ax1.hist(tdeltas.delta_y_classical, bins=nbins, 
            label=f"y ($\sigma$ = {np.std(tdeltas.delta_y_classical):.2f})", alpha=0.7)
    ax1.set_xlabel("Classical (True - Predicted) Positions",fontsize=14)
    ax1.set_ylabel("Counts/bin",fontsize=14)
    ax1.legend()
        
    
def plotxyz(tdl, nbins=50):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    flat_axes = axes.ravel()
    ax0, ax1, ax2 = flat_axes[0], flat_axes[1], flat_axes[2]
    
    ax0.hist(tdl.delta_x_NN, bins=nbins, 
             label=f"x ($\sigma$ = {np.std(tdl.delta_x_NN):.2f})", alpha=0.7)
    ax0.set_xlabel("NN (xtrue - xpredicted)",fontsize=14)
    ax0.set_ylabel("Counts/bin",fontsize=14)
    ax0.legend()
    ax1.hist(tdl.delta_y_NN, bins=nbins, 
             label=f"y ($\sigma$ = {np.std(tdl.delta_y_NN):.2f})", alpha=0.7)
    ax1.set_xlabel("NN (ytrue - ypredicted)",fontsize=14)
    ax1.set_ylabel("Counts/bin",fontsize=14)
    ax1.legend()
    ax2.hist(tdl.delta_z_NN, bins=nbins, 
             label=f"z ($\sigma$ = {np.std(tdl.delta_z_NN):.2f})", alpha=0.7)
    ax2.set_xlabel("NN (ztrue - zpredicted)",fontsize=14)
    ax2.set_ylabel("Counts/bin",fontsize=14)
    ax2.legend()