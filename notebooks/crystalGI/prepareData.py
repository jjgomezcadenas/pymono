import numpy as np
import pandas as pd
import glob

from gi import event_size, fiducial_df, twocluster_df, histoplot, scatter_xy, event_mult, d12, dtz, plot_amplitude
from imgs import plot_image

g4dir = "/Users/jjgomezcadenas/Data/G4Prods/crystalGI/G4BGO"
fid = "isensor_data_bgo_1.csv"
fgi = "gamma_interactions_bgo_1.csv"
fgp = "global_pars.csv"
fsp = "sensor_positions.csv"

csvfiles = glob.glob(f"{csvdir}/*.csv")