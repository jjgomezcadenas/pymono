import os
import glob

from pymono.cnn_xevent import event_size
from pymono.cnn_xevent import filter_and_voxelize_df, mean_XYZ, compute_xyz


def main():
    # Path to monolith data
    path_to_data=os.environ['MONOLITH_DATA'] 
    h5file = "CsI_6x6_fullwrap_50k_0MHzDC_PTFE_LUT_gamma_h5"
    #h5file     = "LYSO_6x6_nowrap_15k_2MHzDC_PTFE_LUT_gamma_NX"
    #h5file = "CsI_6x6_fullwrap_50k_0MHzDC_PTFE_LUT_gamma_2_h5"

    # cutoffs: 10 keV for partices and maximum length of particles in crystal
    eth = 0.001 # in keV
    lmax = 49 # in mm   notice crystal is 50 mm
    vxyz = 12 # in mm

    dir = os.path.join(path_to_data, h5file)
    
    # Define the pattern to match all .txt files in the 'documents' directory
    pattern = f'{dir}/*.h5'

    # Use glob to find all files matching the pattern
    file_paths = glob.glob(pattern)

    # Compute the dimensions of the crystal

    XYZ = compute_xyz(file_paths, start=0, end=5, eth=eth, lmax=lmax, bins=100, prnt=100)
    xyzc = mean_XYZ(XYZ)


    print(f"h5 dir = {h5file}, number of files = {len(file_paths)}")

    ntot, nint, df1c, df2c, dfnc =filter_and_voxelize_df(file_paths, xyzc, start=0, end=len(file_paths), 
                                                         eth=eth, lmax=lmax, sx=vxyz, sy=vxyz, sz=vxyz, prnt=10)
    
    print(f"total events = {ntot}, gamma interact in crystal = {nint}")
    print(f" fraction of interacting events 1g = {(event_size(df1c)/nint):.2f}")
    print(f" fraction of interacting events 2g = {(event_size(df2c)/nint):.2f}")
    print(f" fraction of interacting events >2g = {(event_size(dfnc)/nint):.2f}")

    df1cx = df1c.sort_values(by="event_id")
    df2cx = df2c.sort_values(by="event_id")
    dfncx = dfnc.sort_values(by="event_id")

    print(df1cx.head(10))
    print(df2cx.head(10))
    print(dfncx.head(10))

    f1c = f"df1c_{vxyz}mm.csv"
    f2c = f"df2c_{vxyz}mm.csv"
    fnc = f"dfnc_{vxyz}mm.csv"

    df1cx.to_csv(os.path.join(dir, f1c), index=False)
    df2cx.to_csv(os.path.join(dir, f2c), index=False)
    dfncx.to_csv(os.path.join(dir, fnc), index=False)

if __name__ == "__main__":
    main()

