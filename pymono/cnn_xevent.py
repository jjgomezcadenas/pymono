import pandas as pd
import numpy as np

from collections import namedtuple
from typing import NamedTuple, List


def event_size(df : pd.DataFrame):
    """
    Return the number of events in df
    """
    return len(np.unique(df.event_id))


def event_ekin(df):
    """
    Return the kinetic energies of events in df
    """
    grdf = df.groupby('event_id')
    ekin = grdf['kin_energy'].sum()
    return ekin

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



def concat_df(file_paths, start=0, end=10):
    """
    Concat dfs and drop uneeded columns

    """
    DF =[]
    for file in file_paths[start:end]:
        gammas = pd.read_hdf(file,"MC/particles")
        DF.append(gammas.drop(['initial_momentum_x', 'initial_momentum_y', 'initial_momentum_z',
                 'final_momentum_x', 'final_momentum_y', 'final_momentum_z',
                'final_proc', 'final_t'], axis=1))
                      
    return pd.concat(DF, axis=0)


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


def streams(gdx, i, nprnt):
    """
    Divide DF in streams (1c, 2c, Z2c)

    """
    
    gdf  = gdx.copy()
    grdf = gdf.groupby('event_id')
    gdf  = grdf.agg(#event_id = ('event_id', 'first'),
                etot = ('ebox', 'sum'),
                ngama = ('ebox', 'count')
                  ).reset_index()

    df1g = gdf[gdf.ngama==1]
    df2g = gdf[gdf.ngama==2]
    dfng = gdf[gdf.ngama>2]
    if i%nprnt==0:
        print(f" fraction of events 1g = {(event_size(df1g)/event_size(gdf)):.2f}")
        print(f" fraction of events 2g = {(event_size(df2g)/event_size(gdf)):.2f}")
        print(f" fraction of events >3g = {(event_size(dfng)/event_size(gdf)):.2f}")
    
    return df1g,df2g,dfng

def filter_df(file_paths, start=0, end=10, eth=0.001, lmax=60, prnt=10):
    """
    Filter and reduce the DF
    1. Select events in which gammas interact in crystal
    2. Remove primary gammas
    3. Impose threshold in energy and length
    4. Drop columns not needed, create columns needed for barycenter and rename fields
    5. Concat files
    """

    DF =[]
    
    for i, file in enumerate(file_paths[start:end]):
        if i%prnt == 0:
            print(f"Reading data frame {i}")
        gammas = pd.read_hdf(file,"MC/particles")
        gdf1 = gammas.drop(['initial_momentum_x', 'initial_momentum_y', 'initial_momentum_z',
                 'final_momentum_x', 'final_momentum_y', 'final_momentum_z',
                'final_proc', 'final_t'], axis=1)

        if i%prnt == 0:
            print(f"Select events in which gammas interact in crystal:")
        gdf2 = select_gammas_interact_in_crystal(gdf1)
        
        if i%prnt == 0:
            print(f"Remove primary gammas: ")
        gdf3 = gdf2[gdf2['mother_id'] != 0]

        if i%prnt == 0:
            print(f"Remove particles not coming from primaryC") 
        grdf = gdf3.groupby('event_id')
        gdf4 = grdf.apply(lambda x: x[x['mother_id'] == 1]).reset_index(drop=True)

        if i%prnt == 0:
            print(f"Impose threshold in energy and length") 
        gdf5 = gdf4[(gdf4['kin_energy'] >eth) & (gdf4.length<lmax)]

        if i%prnt == 0:
            print(f"Drop some extra columns and rename fields") 
        gdf6 = gdf5.drop(['final_x', 'final_y', 'final_z', 'length',
                  'primary', 'mother_id', 'initial_volume', 'final_volume'], axis=1)

        gdf6.rename(columns={'initial_x': 'x'}, inplace=True)
        gdf6.rename(columns={'initial_y': 'y'}, inplace=True)
        gdf6.rename(columns={'initial_z': 'z'}, inplace=True)
        gdf6.rename(columns={'kin_energy': 'E'}, inplace=True)

        if i%prnt == 0:
            print(f"Compute xiEi, yiEi, ziEi") 

        gdf6['xE'] = gdf6['x'] * gdf6['E']
        gdf6['yE'] = gdf6['y'] * gdf6['E']
        gdf6['zE'] = gdf6['z'] * gdf6['E']
                
        DF.append(gdf6)

    print(f"now concat:") 
    df = pd.concat(DF, axis=0)

    return df



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




def filter_and_voxelize_df2(file_paths : List[str], mcrst : NamedTuple, start=0, end=10, eth=0.001, lmax=60, sx=6,sy=6,sz=6, prnt=10):
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
        
        #return df1g.sort_values(by="event_id"),df2g.sort_values(by="event_id"),dfng.sort_values(by="event_id")
        return df1g,df2g,dfng
        
   
    Eventt =[]
    Eventg =[]
    DF1c = []
    DF2c = []
    DFnc = []
    
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
            
        Eventt.append(event_size(gdf5))
        gdm = voxelize(gdf5,mcrst,sx,sy,sz)
        gdmx = gdm.drop(['ix', 'iy', 'iz'], axis=1)

        ## Split data into three streams (1c, 2c, > 2c), 
        df1g,df2g,dfng = streams(gdm)
        gdm1c = gdmx[gdmx['event_id'].isin(df1g.event_id.values)]
        gdm2c = gdmx[gdmx['event_id'].isin(df2g.event_id.values)]
        gdmnc = gdmx[gdmx['event_id'].isin(dfng.event_id.values)]


        print("gdm1c")
        print(gdm1c.head(10))
        print("gdm2c")
        print(gdm2c.head(10))
        print("gdmnc")
        print(gdmnc.head(10))

        DF1c.append(gdm1c)
        DF2c.append(gdm2c)
        DFnc.append(gdmnc)

        print(f"total events = {event_size(gammas)}, gamma interact in crystal = {event_size(gdf5)}")
        print(f" fraction of interacting events 1g = {(event_size(gdm1c)/event_size(gdf5)):.2f}")
        print(f" fraction of interacting events 2g = {(event_size(gdm2c)/event_size(gdf5)):.2f}")
        print(f" fraction of interacting events >2g = {(event_size(gdmnc)/event_size(gdf5)):.2f}")
        
    df1c = pd.concat(DF1c, axis=0).sort_values(by="event_id")
    df2c = pd.concat(DF2c, axis=0).sort_values(by="event_id")
    dfnc = pd.concat(DFnc, axis=0).sort_values(by="event_id")
    ntot = np.sum(Eventg)
    nint = np.sum(Eventt)

    print("df1c")
    print(df1c.head(10))
    print("df2c")
    print(df2c.head(10))
    print("dfnc")
    print(dfnc.head(10))

    print(f"eventg->{Eventg[0:10]}")
    print(f"eventt->{Eventt[0:10]}")
    print(f"ntot = {ntot}, nint = {nint}")

    return ntot,nint, df1c, df2c, dfnc

def filter_and_voxelize_df_1c2c(file_paths : List[str], mcrst : NamedTuple, start=0, end=10, eth=0.001, lmax=60, sx=6,sy=6,sz=6, prnt=10):
    """
    Reduce and filter the data frame, then voxelize i considering only two type of events
    
    """
    
    def event_size(df):
        return len(np.unique(df.event_id))
    
    
    def voxelize(gdfx,mcrst,sx,sy,sz):
        """
        Voxelize the crystal and compute x,y,z,e, & t in each box
        
        """
    
        gdf = gdfx.copy()
        
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
        df2g = gdf[gdf.ngama>=2]
        #dfng = gdf[gdf.ngama>2]
        
        return df1g,df2g
        #return df1g,df2g,dfng
        
        
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

    #Eventnc =[]
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
        #  Stream 2g: events with 2 or more clusters: true information: x1,y1,z1,t1,e1, x2,y2,z2,t2,e2
        #df1g,df2g,dfng = streams(gdm)
        df1g,df2g = streams(gdm)

        gdmx = gdm.drop(['ix', 'iy', 'iz'], axis=1)
        gdm1c = gdmx[gdmx['event_id'].isin(df1g.event_id.values)]
        gdm2c = gdmx[gdmx['event_id'].isin(df2g.event_id.values)]
        #gdmnc = gdmx[gdmx['event_id'].isin(dfng.event_id.values)]

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

        #Eventnc.extend(gdmnc.event_id.values) 
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

    # data = {
    # 'event_id': np.unique(np.array(Eventnc))}
    # dfnc   = pd.DataFrame(data)
    
    return np.sum(Eventg), np.sum(Eventt), df1c, df2c
    #return np.sum(Eventg), np.sum(Eventt), df1c, df2c, dfnc



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
        #if i%prnt ==0:
        #    print(f"Reading data frame {i}")
        gammas = pd.read_hdf(file,"MC/particles")

    
        gdf1 = select_gammas_interact_in_crystal(gammas)
        gdf2 = gdf1[gdf1['mother_id'] != 0]

        #if i%prnt ==0:
            #print(f"Remove particles not coming from primaryC") 
        grdf = gdf2.groupby('event_id')
        gdf3 = grdf.apply(lambda x: x[x['mother_id'] == 1]).reset_index(drop=True)
        gdf4 = gdf3[(gdf3['kin_energy'] >eth) & (gdf3.length<lmax)]

        FillXYZ(gdf4, XYZ)

    return XYZ
