import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def select_image_from_df(df,evtsel, n = 8):
    df2=df[df.event==evtsel]
    
    charge_matrix = np.zeros((n, n))
    sensor_id = df2['sensor_id'].values
    charge    = df2['amplitude'].values

    for id in sensor_id:
        charge_matrix[sensor_id[id] // n, sensor_id[id] % n] = charge[id]
    return charge_matrix


def get_gamma_position(dfg, evtsel, x_spatial, y_spatial):
    df = dfg[dfg.event==evtsel]
    print(f"xg1 = {df.x1.values[0]}, yg1 ={df.y1.values[0]}")
    
    
    xt1, yt1 = transform_coordinates(df.x1.values[0], df.y1.values[0], 
                                     x_spatial, y_spatial)

    print(f"xt1 = {xt1}, yt1 ={yt1}")

    print(f"xg2 = {df.x1.values[0]}, yg2 ={df.y1.values[0]}")

    xt2, yt2 = transform_coordinates(df.x2.values[0], df.y2.values[0], 
                                     x_spatial, y_spatial)
    print(f"xt2 = {xt2}, yt1 ={yt2}")

    return xt1, yt1,xt2, yt2


def transform_coordinates(x, y, x_spatial, y_spatial, x_min2=0,   x_max2=8,  y_min2=0,   y_max2 = 8):
                        
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

    print(f"x = {x}, y ={y}")
  

    return x_new, y_new


def plot_image(dfq, dfg,  evtsel, x_spatial, y_spatial, figsize=(6, 6)):
    
    image = select_image_from_df(dfq,evtsel)
    plot_image2(image, dfg,  evtsel, x_spatial, y_spatial, figsize=figsize)
    

def plot_image2(image, dfg,  evtsel, x_spatial, y_spatial, figsize=(6, 6)):
    
    charge_matrix = image
    xt1, yt1,xt2, yt2 =get_gamma_position(dfg, evtsel, x_spatial, y_spatial)
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot the image with imshow (with pixel axis from 0 to 8)
    img = ax1.imshow(charge_matrix.T, extent=[0, 8, 0, 8], origin='lower', aspect='auto',
                    cmap='viridis', interpolation='none')

    # Add colorbar for the imshow plot
    cbar = fig.colorbar(img, ax=ax1, pad=0.18)
    cbar.set_label('Charge')

    # Create a secondary x-axis (spatial) that matches the pixel axis
    ax2 = ax1.twiny()
    ax2.set_xlim(x_spatial[0], x_spatial[-1])
    ax2.set_xlabel('X')

    # Create a secondary y-axis (spatial) that matches the pixel axis
    ax3 = ax1.twinx()
    ax3.set_ylim(y_spatial[0], y_spatial[-1])
    ax3.set_ylabel('Y')

    # Show the plot

    ax1.scatter(xt1, yt1,  facecolor='red')
    ax1.scatter(xt2, yt2,  facecolor='blue')

    ax1.text(2, 6, f"Event = {evtsel}", color='white', fontsize=12, ha='center', va='center')

    plt.tight_layout()
    plt.show()