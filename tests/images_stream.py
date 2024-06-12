#!/usr/bin/env python

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os


def write_images(events, ifolder, ofolder, 
                 events_per_file, got_positions, crystal_width, sipm_width, h5fpfx):
   
    events["file_id"] = (events["event_id"] // events_per_file).astype(int)

    for group_name, group_data in tqdm(events.groupby("file_id")):
        nexus_file = f"{ifolder}/{h5fpfx}.{group_name}.h5"
        print(f"reading nexus file = {nexus_file}")

        if not got_positions:
            monolithic_csi_positions = pd.read_hdf(nexus_file, "MC/sns_positions")
            sensor_ids = np.array(monolithic_csi_positions['sensor_id'])
            pos_dict = {}
            for sensor_id in sensor_ids:
                pos_dict[sensor_id] = {'x': int((monolithic_csi_positions.query(f"sensor_id=={sensor_id}")['x'].iloc(0)[0]+crystal_width/2)/sipm_width),
                                       'y': int((monolithic_csi_positions.query(f"sensor_id=={sensor_id}")['y'].iloc(0)[0]+crystal_width/2)/sipm_width)}
            got_positions = True

        sns_response = pd.read_hdf(nexus_file, "MC/sns_response")
        this_image_events = group_data["event_id"] 
        sns_response = sns_response[sns_response["event_id"].isin(this_image_events)]

        events = np.unique(sns_response['event_id'])
        images = np.zeros((events.shape[0],crystal_width//sipm_width,crystal_width//sipm_width))
        ev_wfs = sns_response.groupby("event_id")
        j = 0
        for _, ev_wf in ev_wfs:
            average_position = ev_wf.groupby("sensor_id").sum()
            for sid in average_position.index:
                position_x = pos_dict[sid]['x']
                position_y = pos_dict[sid]['y']
                images[j, position_x, position_y] += average_position['charge'][sid]
            j += 1

        print(f"saving file = {ofolder}/images_{group_name}.npy")

        np.save(f"{ofolder}/images_{group_name}.npy", images.astype(np.float32))


def get_folder_out(folder_h5, voxelization):
    ns = folder_h5.split("_")[:-1] # remove "h5" or "NX"
    ns.append("vox")
    ns.append(voxelization) # add "vox_{voxelization}" to the tail of the string
    return '_'.join(ns) 

def main():
    
    path_to_data =os.environ['MONOLITH_DATA'] # Path to monolith data
    folder_h5     = "CsI_6x6_fullwrap_50k_0MHzDC_PTFE_LUT_gamma_h5" # input h5 folder
    #folder_h5     = "LYSO_6x6_nowrap_15k_2MHzDC_PTFE_LUT_gamma_NX"
    #folder_h5      =  "CsI_6x6_fullwrap_50k_0MHzDC_PTFE_LUT_gamma_2_h5"
    h5fpfx         = "MonolithicCsI.CsI" # prefix of files
    #h5fpfx        = "MonolithicCsI.LYSO" # prefix of files
    

    vxy = 12 # in mm
    vz = 12 # in mm

    voxelization  = f"xy_{vxy}_z_{vz}"
    clx ="df2c"


    events_per_file = 10000
    got_positions = False
    crystal_width = 48
    sipm_width = 6
    
    ifolder = os.path.join(path_to_data, folder_h5)
    ofolder = os.path.join(path_to_data, get_folder_out(folder_h5, voxelization), clx)   
    category = f"{clx}_{voxelization}"
    idf = f"{category}.csv"

    print(f"input folder ={ifolder}, output folder = {ofolder}")
    print(f"input DF ={idf}")
    print(f"voxelization ={voxelization}")
    print(f"class ={clx}")

    # Create output directory if it doesn't exist
    
    if not os.path.exists(ofolder):
        os.makedirs(ofolder)
    
    events = pd.read_csv(os.path.join(ifolder, idf), delimiter=",")
    write_images(events, ifolder, ofolder,
                 events_per_file, got_positions, crystal_width, sipm_width, h5fpfx)


if __name__ == "__main__":
    main()
