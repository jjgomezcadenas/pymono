import matplotlib.pyplot as plt
import numpy as np 
from torchvision import transforms

def plot_images2(imgs, dfs, img_range, pixel_size = 6, grid_size=8):
# TODO: this should be called plot_images1c

    fig, axs = plt.subplots(2, 4,figsize=(14, 4))        
    ftx = axs.ravel()
    for i, ev in enumerate(range(*img_range)):  
        dfi = dfs.iloc[ev]
        x_evt = (dfi['y'] + pixel_size*grid_size/2)/pixel_size - 0.5
        y_evt = (dfi['x'] + pixel_size*grid_size/2)/pixel_size - 0.5
       
        #print(f"ev = {ev}, event_id = {evt}, x = {x}, y = {y} xevt = {x_evt}, yevt = {y_evt}")
        
        ftx[i].imshow(imgs[ev,:,:])
        ftx[i].plot([x_evt],[y_evt],'o',color='red')


def plot_images2c(imgs, dfs, img_range, pixel_size = 6, grid_size=8):
    """
    imshow plots the img^T, thus, x swapped by y just for plotting
    """
    fig, axs = plt.subplots(2, 4,figsize=(14, 4))        
    ftx = axs.ravel()
    for i, ev in enumerate(range(*img_range)):  
        dfi = dfs.iloc[ev]
        x1_evt = (dfi['y1'] + pixel_size*grid_size/2)/pixel_size - 0.5
        y1_evt = (dfi['x1'] + pixel_size*grid_size/2)/pixel_size - 0.5
        x2_evt = (dfi['y2'] + pixel_size*grid_size/2)/pixel_size - 0.5
        y2_evt = (dfi['x2'] + pixel_size*grid_size/2)/pixel_size - 0.5
        bx     = (dfi['y1'] * dfi["e1"] + dfi['y2'] * dfi["e2"]) / dfi["etot"]
        by     = (dfi['x1'] * dfi["e1"] + dfi['x2'] * dfi["e2"]) / dfi["etot"]
        bx_evt = (by + pixel_size*grid_size/2)/pixel_size - 0.5
        by_evt = (bx + pixel_size*grid_size/2)/pixel_size - 0.5
        
        ftx[i].imshow(imgs[ev,:,:])
        ftx[i].plot([x1_evt],[y1_evt],'o',color='red')
        ftx[i].plot([x2_evt],[y2_evt],'o',color='blue')
        ftx[i].plot([bx_evt],[by_evt],'o',color='green')


def plot2c_z(dfs, img_range, figsize=(4, 8)):
    """
    z positions
    """
    fig, ftx = plt.subplots(1, 2,figsize=figsize)        
    z1 = []
    z2 = []
    for i, ev in enumerate(range(*img_range)):  
        dfi = dfs.iloc[ev]
        z1.append(dfi['z1']) 
        z2.append(dfi['z2']) 
    zz = np.array(z1) - np.array(z2)
    ftx[0].plot([z1],[z2],'o',markersize=3, color='red')
    ftx[0].set_ylabel("z2")
    ftx[0].set_xlabel("z1")
    ftx[1].hist(zz, bins=50)
    ftx[1].set_ylabel("Entries")
    ftx[1].set_xlabel("z1-z2")
    fig.tight_layout()
    plt.show()


def plot_images_and_labels(train_loader, start=0, figsize=(10, 8)):
    IMG = []
    LBL = []

    for i in range(0,start):
        train_features, train_labels = next(iter(train_loader))
    
    for i in range(0,9):
        train_features, train_labels = next(iter(train_loader))
    
        IMG.append(train_features[0].squeeze())
        LBL.append(train_labels[0])
        
    fig, axs = plt.subplots(3, 3,figsize=figsize)        
    ftx = axs.ravel()
    for  i in range(0,9):        
        img = IMG[i]
        lbl = LBL[i]
        ftx[i].imshow(transforms.ToPILImage()(img))
        ftx[i].text(2, 2, f'{lbl}', color='red', ha='center') 
        #ftx[i].title("label")
    plt.show()


def plot_images_and_positions(train_loader, start=0, pixel_size = 6, grid_size=8, figsize=(10, 8)):
    IMG = []
    LBL = []

    for i in range(0,start):
        train_features, train_labels = next(iter(train_loader))
    
    for i in range(0,9):
        train_features, train_labels = next(iter(train_loader))
    
        IMG.append(train_features[0].squeeze())
        LBL.append(train_labels[0])
        
    _, axs = plt.subplots(3, 3,figsize=figsize)        
    ftx = axs.ravel()
    for  i in range(0,9):        
        img = IMG[i]
        xyz = LBL[i]
        x_evt = (xyz[1] + pixel_size*grid_size/2)/pixel_size - 0.5
        y_evt = (xyz[0] + pixel_size*grid_size/2)/pixel_size - 0.5
        ftx[i].imshow(transforms.ToPILImage()(img))
        ftx[i].plot([x_evt],[y_evt],'o',color='red')
    plt.show()

