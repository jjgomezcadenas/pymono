import pandas as pd
import numpy as np

import os
import tqdm
from collections import namedtuple

import glob

from typing import NamedTuple, List


def event_size(df : pd.DataFrame):
    """
    Return the number of events in df
    """
    return len(np.unique(df.event_id))



def select_gammas_interact_in_crystal(df):
    grdf = df.groupby('event_id')
    return grdf.filter(lambda x: ((x.mother_id == 0) & (x['final_volume'] == 'CRYSTAL')).any() )


def xindex(x, mcrst, sx):
    """
    Given a coordinate x in a detector of size specified by mcrst
    with voxelization sx, return the corresponding index of voxel. 

    """
    return np.floor((x + mcrst.xmin + mcrst.dx)/sx)

def yindex(x,mcrst, sy):
    """
    Given a coordinate y in a detector of size specified by mcrst
    with voxelization sy, return the corresponding index of voxel. 
    
    """
    return np.floor((x + mcrst.ymin + mcrst.dy)/sy)

def zindex(x,mcrst, sz):
    """
    Given a coordinate z in a detector of size specified by mcrst
    with voxelization sz, return the corresponding index of voxel. 
    
    """
    return np.floor((x - mcrst.zmin)/sz)


def voxelize(gdfx,mcrst,i, sx,sy,sz,case='mono', prnt=10):
    """
    Voxelize dataframe gdfx along two (pixel) or three axis (mono)

    """

    gdf = gdfx.copy()
    nx = int(mcrst.dx/sx)
    ny = int(mcrst.dy/sy)
    nz = int(mcrst.dz/sz)
    
    if i%prnt == 0:
        print(f"nx = {nx}, ny = {ny}, nz = {nz}")
        print(f"number of voxels = {nx*ny*nz}")
        print(f"index for xmin ={xindex(mcrst.xmin, mcrst, sx)} index for xmax ={xindex(mcrst.xmax, mcrst, sx)}")
        print(f"index for ymin ={yindex(mcrst.ymin, mcrst, sy)} index for ymax ={yindex(mcrst.ymax, mcrst, sy)}")
        print(f"index for zmin ={zindex(mcrst.zmin, mcrst, sz)} index for zmax ={zindex(mcrst.zmax, mcrst, sz)}")

    gdf["ix"]= xindex(gdf.x, mcrst, sx).astype(int)
    gdf["iy"]= yindex(gdf.y, mcrst, sy).astype(int)
    gdf["iz"]= zindex(gdf.z, mcrst, sz).astype(int)

    if case == 'pixel':
        grdf = gdf.groupby(['event_id', 'ix', 'iy'])
    else:
         grdf = gdf.groupby(['event_id', 'ix', 'iy', 'iz'])

    # prepare DF to compute baricenter: add columns with xiEi
    gdf = grdf.agg(
                ebox = ('E', 'sum'),
                tbox = ('initial_t', 'first'),     
                x1 = ('x', 'first'),     
                y1 = ('y', 'first'),        
                z1 = ('z', 'first'),        # z in the box is the y mean
                xbox = ('xE', 'sum'),     
                ybox = ('yE', 'sum'),        
                zbox = ('zE', 'sum'),        # z in the box is the y mean
                nbox = ('x', 'count'),  # this counts the number of occurences
                  ).reset_index()
    gdf['xbox'] = gdf['xbox'] / gdf['ebox']
    gdf['ybox'] = gdf['ybox'] / gdf['ebox']
    gdf['zbox'] = gdf['zbox'] / gdf['ebox']
    return gdf


def filter_and_voxelize_df(file_paths : List[str], mcrst : NamedTuple, start=0, end=10, eth=0.001, lmax=60, sx=6,sy=6,sz=6, prnt=10):
    """
    Reduce and filter the data frame, then voxelize it. 
    
    """
    
    def event_size(df):
        return len(np.unique(df.event_id))
    
    
    def voxelize(gdfx,mcrst,sx,sy,sz):
        """
        Voxelize the crystal and compute x,y,z,e, & t in each box
        
        """
    
        gdf = gdfx.copy()
        #nx = int(mcrst.dx/sx)
        #ny = int(mcrst.dy/sy)
        #nz = int(mcrst.dz/sz)
        
        gdf["ix"]= xindex(gdf.initial_x, mcrst, sx).astype(int)
        gdf["iy"]= yindex(gdf.initial_y, mcrst, sy).astype(int)
        gdf["iz"]= zindex(gdf.initial_z, mcrst, sz).astype(int)
    
        grdf = gdf.groupby(['event_id', 'ix', 'iy', 'iz'])

        # Elements in the box
        gdf = grdf.agg(
                    e = ('kin_energy', 'sum'),
                    t = ('initial_t', 'first'),     
                    x = ('initial_x', 'first'),     
                    y = ('initial_y', 'first'),        
                    z = ('initial_z', 'first'),        # z in the box is the y mean
                      ).reset_index()
        
        return gdf
        

    def streams(gdx):
        """
        Divide the data in streams
        
        """
    
        gdf  = gdx.copy()
        grdf = gdf.groupby('event_id')
        gdf  = grdf.agg(#event_id = ('event_id', 'first'),
                    ngama = ('e', 'count')
                      ).reset_index()
    
        df1g = gdf[gdf.ngama==1]
        df2g = gdf[gdf.ngama==2]
        dfng = gdf[gdf.ngama>2]
        
        return df1g,df2g,dfng
        
    Event1c =[] 	
    E = []
    X = []
    Y = []
    Z = []
    T = []

    Event2c =[] 	
    E1 = []
    X1 = []
    Y1 = []
    Z1 = []
    T1 = [] 

    E2 = []
    X2 = []
    Y2 = []
    Z2 = []
    T2 = [] 
    ET2 = []

    Eventnc =[]
    Eventt =[]
    Eventg =[]
        
    
    for i, file in enumerate(file_paths[start:end]):
        if i%prnt == 0:
            print(f"Reading data frame {i}")
        gammas = pd.read_hdf(file,"MC/particles")

        Eventg.append(event_size(gammas))
        
        gammas.drop(['initial_momentum_x', 'initial_momentum_y', 'initial_momentum_z',
                 'final_momentum_x', 'final_momentum_y', 'final_momentum_z',
                'final_proc', 'final_t'], axis=1)

        gdf1 = select_gammas_interact_in_crystal(gammas)
        gdf2 = gdf1[gdf1['mother_id'] != 0]
        grdf = gdf2.groupby('event_id')
        gdf3 = grdf.apply(lambda x: x[x['mother_id'] == 1]).reset_index(drop=True)
        gdf4 = gdf3[(gdf3['kin_energy'] >eth) & (gdf3.length<lmax)]
        gdf5 = gdf4.drop(['final_x', 'final_y', 'final_z', 'length',
                  'primary', 'mother_id', 'initial_volume', 'final_volume'], axis=1)
            
        gdm = voxelize(gdf5,mcrst,sx,sy,sz)

        ## Split data into three streams.
        #  Stream 1g: events with 1 cluster: true information: x,y,z,t,e
        #  Stream 2g: events with 2 clusters: true information: x1,y1,z1,t1,e1, x2,y2,z2,t2,e2
        #  Stream ng: events with more than 2 clusters: no true information needed
        df1g,df2g,dfng = streams(gdm)

        gdmx = gdm.drop(['ix', 'iy', 'iz'], axis=1)
        gdm1c = gdmx[gdmx['event_id'].isin(df1g.event_id.values)]
        gdm2c = gdmx[gdmx['event_id'].isin(df2g.event_id.values)]
        gdmnc = gdmx[gdmx['event_id'].isin(dfng.event_id.values)]

        ## Two cluster case
        grdf = gdm2c.groupby('event_id')
        gdy2ca = grdf.agg(#event_id = ('event_id', 'first'),
                etot = ('e', 'sum'),
                e1 = ('e', 'first'),
                e2 = ('e', 'last'),
                x1 = ('x', 'first'),
                x2 = ('x', 'last'),
                y1 = ('y', 'first'),
                y2 = ('y', 'last'),
                z1 = ('z', 'first'),
                z2 = ('z', 'last'),
                t1 = ('t', 'first'),
                t2 = ('t', 'last')
                  ).reset_index()
        
        Event1c.extend(gdm1c.event_id.values) 
        E.extend(gdm1c.e.values) 	
        X.extend(gdm1c.x.values)
        Y.extend(gdm1c.y.values)
        Z.extend(gdm1c.z.values)
        T.extend(gdm1c.t.values)

        Event2c.extend(gdy2ca.event_id.values) 	
        E1.extend(gdy2ca.e1.values) 	
        X1.extend(gdy2ca.x1.values)
        Y1.extend(gdy2ca.y1.values)
        Z1.extend(gdy2ca.z1.values)
        T1.extend(gdy2ca.t1.values)
        E2.extend(gdy2ca.e2.values) 	
        X2.extend(gdy2ca.x2.values)
        Y2.extend(gdy2ca.y2.values)
        Z2.extend(gdy2ca.z2.values)
        T2.extend(gdy2ca.t2.values)
        ET2.extend(gdy2ca.etot.values) 

        Eventnc.extend(gdmnc.event_id.values) 
        Eventt.append(event_size(gdf4))


    # Create composed dfs
    data = {
    'event_id': Event1c,
    'e': E,
    'x': X,
    'y': Y,
    'z': Z,
    't': T
    }
    df1c   = pd.DataFrame(data)

    data = {
    'event_id': Event2c,
    'e1': E1,
    'x1': X1,
    'y1': Y1,
    'z1': Z1,
    't1': T1,
    'e2': E2,
    'x2': X2,
    'y2': Y2,
    'z2': Z2,
    't2': T2,
    'etot': ET2
    }
    df2c   = pd.DataFrame(data)

    data = {
    'event_id': np.unique(np.array(Eventnc))}
    dfnc   = pd.DataFrame(data)
    
    return np.sum(Eventg), np.sum(Eventt), df1c, df2c, dfnc


def mean_XYZ(XYZ):
    XMIN = []
    XMAX = []
    YMIN = []
    YMAX = []
    ZMIN = []
    ZMAX = []
    for xyz in XYZ:
        XMIN.append(xyz.xmin)
        XMAX.append(xyz.xmax)
        YMIN.append(xyz.ymin)
        YMAX.append(xyz.ymax)
        ZMIN.append(xyz.zmin)
        ZMAX.append(xyz.zmax)
        
    xmin = np.ceil(np.mean(np.array(XMIN)))
    xmax = np.floor(np.mean(np.array(XMAX)))
    ymin = np.ceil(np.mean(np.array(YMIN)))
    ymax = np.floor(np.mean(np.array(YMAX)))
    zmin = np.ceil(np.mean(np.array(ZMIN)))
    zmax = np.floor(np.mean(np.array(ZMAX)))
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    
    MCrst = namedtuple('MCrst','xmin xmax dx ymin ymax dy zmin zmax dz')
    return MCrst(xmin, xmax, dx, ymin, ymax, dy, zmin, zmax, dz)


def compute_xyz(file_paths, start=0, end=10, eth=0.001, lmax=45, bins=100,prnt=10):
    """
    Compute the dimensions of the crystal from the data themselves
    
    """

    def select_gammas_interact_in_crystal(df):
        grdf = df.groupby('event_id')
        return grdf.filter(lambda x: ((x.mother_id == 0) & (x['final_volume'] == 'CRYSTAL')).any() )
    

    def xyz(h):
        zmin = np.ceil(h[0])
        zmax = np.floor(h[-1])
        dz = zmax - zmin
        return zmin, zmax, dz
    

    def FillXYZ(df, XYZ):               
        _, b = np.histogram(df.initial_z, bins)
        zmin, zmax, dz = xyz(b)          
        
        _, b = np.histogram(df.initial_x, bins)
        xmin, xmax, dx = xyz(b)          
        
        _, b = np.histogram(df.initial_y, bins)
        ymin, ymax, dy = xyz(b)          
        
        XYZ.append(MCrst(xmin, xmax, dx, ymin, ymax, dy, zmin, zmax, dz))


    XYZ = []
    MCrst = namedtuple('MCrst','xmin xmax dx ymin ymax dy zmin zmax dz')
    
    for i, file in enumerate(file_paths[start:end]):
        if i%prnt ==0:
            print(f"Reading data frame {i}")
        gammas = pd.read_hdf(file,"MC/particles")

    
        gdf1 = select_gammas_interact_in_crystal(gammas)
        gdf2 = gdf1[gdf1['mother_id'] != 0]

        if i%prnt ==0:
            print(f"Remove particles not coming from primaryC") 
        grdf = gdf2.groupby('event_id')
        gdf3 = grdf.apply(lambda x: x[x['mother_id'] == 1]).reset_index(drop=True)
        gdf4 = gdf3[(gdf3['kin_energy'] >eth) & (gdf3.length<lmax)]

        FillXYZ(gdf4, XYZ)

    return XYZ


def main():
    # Path to monolith data
    path_to_data=os.environ['MONOLITH_DATA'] 
    h5file = "CsI_6x6_fullwrap_50k_0MHzDC_PTFE_LUT_gamma_h5"

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

    ntot, nint, df1c, df2c, dfnc =filter_and_voxelize_df(file_paths, xyzc, start=0, end=100, 
                                                         eth=eth, lmax=lmax, sx=vxyz, sy=vxyz, sz=vxyz, prnt=10)
    
    print(f"total events = {ntot}, gamma interact in crystal = {nint}")
    print(f" fraction of interacting events 1g = {(event_size(df1c)/nint):.2f}")
    print(f" fraction of interacting events 2g = {(event_size(df2c)/nint):.2f}")
    print(f" fraction of interacting events >2g = {(event_size(dfnc)/nint):.2f}")
    print(df1c.head(10))
    print(df2c.head(10))
    print(dfnc.head(10))

    f1c = f"df1c_{vxyz}mm.csv"
    f2c = f"df2c_{vxyz}mm.csv"
    fnc = f"dfnc_{vxyz}mm.csv"

    df1c.to_csv(os.path.join(path_to_data, f1c), index=False)
    df2c.to_csv(os.path.join(path_to_data, f2c), index=False)
    dfnc.to_csv(os.path.join(path_to_data, fnc), index=False)

if __name__ == "__main__":
    main()

