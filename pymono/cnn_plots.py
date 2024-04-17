import matplotlib.pyplot as plt
import numpy as np 
from torchvision import transforms

def plot_images2(imgs, dfs, img_range, pixel_size = 6, grid_size=8):

    fig, axs = plt.subplots(2, 4,figsize=(14, 4))        
    ftx = axs.ravel()
    for i, ev in enumerate(range(*img_range)):  
        dfi = dfs.iloc[ev]
        x_evt = (dfi['y'] + pixel_size*grid_size/2)/pixel_size - 0.5
        y_evt = (dfi['x'] + pixel_size*grid_size/2)/pixel_size - 0.5
       
        #print(f"ev = {ev}, event_id = {evt}, x = {x}, y = {y} xevt = {x_evt}, yevt = {y_evt}")
        
        ftx[i].imshow(imgs[ev,:,:])
        ftx[i].plot([x_evt],[y_evt],'o',color='red')


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
