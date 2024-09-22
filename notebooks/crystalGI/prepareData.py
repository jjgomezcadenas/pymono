import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import time

from gi import event_size, fiducial_df, twocluster_df, histoplot, scatter_xy, event_mult, d12, dtz, plot_amplitude
from imgs import  plot_image2


def get_files(csvdir):
    csvfiles = glob.glob(f"{csvdir}/*.csv")
    gamma_files = [file  for file in csvfiles if file.find("gamma_")>0]
    sensor_files = [file  for file in csvfiles if file.find("isensor_")>0]
    global_files = [file  for file in csvfiles if file.find("global")>0]
    return gamma_files, sensor_files, global_files
    

def get_file_number(xf): 
    return int(xf.split("/")[-1].split("_")[-1].split(".")[0])
    

def sort_files(unsorted_files, sorting_vector):
    sorted_files = list(unsorted_files)
    for i, f in enumerate(unsorted_files):
        indx = sorting_vector[i] - 1
        sorted_files[indx]=f
    return sorted_files
    
    
def collect_images(df, n=8):
    events = np.unique(df['event'])
    images = np.zeros((events.shape[0],n,n))
    gevt = df.groupby('event')
    i=0
    charge_matrix = np.zeros((n, n))
    for event_number, group in gevt:
        for _, row in group.iterrows():
            sensor_id = row['sensor_id']
            charge = row['amplitude']
            charge_matrix[sensor_id // n, sensor_id % n] = charge
        images[i]= charge_matrix
        i+=1
    return images
    

def create_image_file(dfg, imgf):
    """
    1. get the image
    2. save as npy in images dir
    """
    
    images = collect_images(dfg)
    np.save(imgf, images)
    

def process_files(csvdir, imgdir):
    """
    Main driver:
    1. read files from the csvdir (e.g, files produced from geant4). These are:
      a) gamma_files with the information of gamma interaction in the detector
      b) sensor_files with the information of number of photoelectrons readout by the SiPMs.
      c) global_files containing metadata (e.g, type of crystal, dimensions, number of events)

    2. Files are sorted

    3. A loop on files reads out the three types of files and extracts images and metadata
    
    """
    gamma_files, sensor_files, global_files = get_files(csvdir)
    gfn = [get_file_number(gf) for gf in global_files]
    gxfn = [get_file_number(gf) for gf in gamma_files]
    sfn = [get_file_number(gf) for gf in sensor_files]
    sorted_gamma_files = sort_files(gamma_files, gxfn)
    sorted_global_files = sort_files(global_files, gfn)
    sorted_sensor_files = sort_files(sensor_files, sfn)
    
    #print(sorted_global_files)
    #print(sorted_gamma_files)
    #print(sorted_sensor_files)

    for i, gf in enumerate(sorted_global_files):
        print(f"\n+++reading {gf}")
        
        dfgp = pd.read_csv(gf, header=0)
        cw = dfgp.crystalWidth.values[0]
        cl = dfgp.crystalLength.values[0]
        nn = dfgp.numberOfEvents.values[0]
        ge = dfgp.gammaEnergy.values[0]
        mat = dfgp.material.values[0]
        sxy = dfgp.sipmXY[0]
        npx = cw//sxy
        print(f"- material = {mat}, crystal width = {cw}, length = {cl} nof = {nn}, npixels ={npx} x {npx}")

        print(f"\n++reading {sorted_gamma_files[i]}")
        
        dfgi = pd.read_csv(sorted_gamma_files[i], header=0).sort_values(by='event')
        print(f"-first event of gamma interaction df {dfgi.iloc[0].event}")
        print(f"-last event of gamma interaction df {dfgi.iloc[-1].event}")

        x_int = event_size(dfgi)/nn
        print(f"-fraction of events interacting in crystal = {x_int}")

        print(f"\n+++fiducial cut")
        dfx = fiducial_df(dfgi, d=cw, z=cl)

        print(f"-mean of x : {(np.mean(dfx.x.values)):.2f}")
        print(f"-mean of y : {np.mean(dfx.y.values):.2f}")
        print(f"-mean of z : {np.mean(dfx.z.values):.2f}")

        ec = 0.98 *ge*1000 # in keV
        print(f"+++cutting on energy: cutoff energy = {ec:.2f}")

        dfe = dfx[dfx.etot>ec]

        x_e = event_size(dfe)/nn
        print(f"-fraction of events with e > {ec:.1f} keV = {x_e}")

        print(f"+++2c data frame")
        df2c = twocluster_df(dfe).drop('index', axis=1)

        dfpe = df2c[df2c.ntrk==1]
        dfco = df2c[df2c.ntrk>1]
        print(f"-fraction of photoelectric events  = {event_size(dfpe)/event_size(df2c)}, of compton = {event_size(dfco)/event_size(df2c)}")
        print(f"-wrt total interactions  = {event_size(dfpe)/nn}, compton = {event_size(dfco)/nn}")

        print(f"+++read sensor data")
        dfgi = pd.read_csv(sorted_sensor_files[i], header=0).sort_values(by='event')

        print(f"-select events that pass cuts in df2c")
        sel_gi = df2c.event.values
        dfi = dfgi[dfgi['event'].isin(sel_gi)]
        
        print(f"-fraction of filtered events ={event_size(dfi)/event_size(dfgi):.2f}")
        print(f"-relative size of gamma and sensor DF (must be one) = {event_size(dfi)/event_size(df2c)}")

        imgf = f"{imgdir}/images_{i}.npy"
        print(f"+++creating image = {imgf}+++")
        create_image_file(dfi, imgf)

        metaf = f"{imgdir}/metadata_{i}.csv"
        print(f"+++creating metadata file = {metaf}+++")
        df2c.to_csv(metaf, index=False)
    

def main():
    start_time = time.time() 
    g4dir = "/Users/jjgomezcadenas/Data/G4Prods/crystalGI/G4BGO"
    imgdir = "/Users/jjgomezcadenas/Data/G4Prods/crystalGI/BGO"
  
    print(f"Input directory (csv files produced by G4) = {g4dir}")
    print(f"Output directory (img and metadata files) = {imgdir}")
    
    process_files(g4dir, imgdir)

    end_time = time.time()  # Stop the clock

    execution_time = end_time - start_time
    print(f"Finished: Execution time: {execution_time:.2f} seconds")

# Main program entry point
if __name__ == "__main__":
    main()
