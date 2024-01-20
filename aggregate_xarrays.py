import xarray 
import dask 
import numpy as np
import os 
store_dir = os.environ['STORE']
scratch_dir = os.environ['SCRATCH']


def convert_to_joined_numpy(year,variation):

    Lmon =  xarray.open_mfdataset(f'{store_dir}/s{year}-r{variation}i1p1f1/Lmon/*.nc',compat='minimal')
    Amon =  xarray.open_mfdataset(f'{store_dir}/s{year}-r{variation}i1p1f1/Amon/*.nc',compat='minimal')
    Emon =  xarray.open_mfdataset(f'{store_dir}/s{year}-r{variation}i1p1f1/Emon/*.nc',compat='override')
    Lmon = Lmon.drop_vars(['time_bounds','depth_bounds','sector','depth'])
    Amon = Amon.drop_vars(['time_bounds'])
    Emon = Emon.drop_vars(['height','depth','tdps','ppdiat','ppmisc','expfe','olevel_bounds','olevel','flandice','t20d','thetaot','thetaot2000','thetaot300','thetaot700','bounds_nav_lat','bounds_nav_lon','lev','area','type'])
    Emon = Emon.drop_dims(['landuse','x','y','axis_nbounds'])
    # variable_list = list(Lmon.keys()) + list(Amon.keys()) + list(Emon.keys())
    # lon,lat = np.meshgrid(Lmon.lon.data,Lmon.lat.data)
    # Lmon['time'] = Lmon.time.astype('float')
    # Amon['time'] = Amon.time.astype('float')
    
    #to convert back, Lmon['time'].astype('datetime64[ns]')
    # time = np.broadcast_to(Lmon.coords['time'],(1,lon.shape[1],lon.shape[2]))
    # time = np.broadcast_to(Lmon.coords['time'],(lon.shape[0],lon.shape[1],len(Lmon.time)))
    # lon = np.broadcast_to(lon,(1,len(Lmon.time),lon.shape[0],lon.shape[1]))
    # lat = np.broadcast_to(lat,(1,len(Lmon.time),lat.shape[0],lat.shape[1]))
    # time = time.reshape(time.shape[2],time.shape[0],time.shape[1])
    # time = time[np.newaxis,:,:,:]
    # total_var_numpy_array = np.concatenate([Lmon.to_array().to_numpy(),Amon.to_array().to_numpy(),Emon.to_array().to_numpy(),lon,lat,time])
    xarray.merge([Lmon,Amon,Emon]).to_netcdf(f'{scratch_dir}/{year}_{variation}.nc')
    # np.save(f'{scratch_dir}/{year}_{variation}',total_var_numpy_array)
    
    
    
    
    #sanity check distributions
#ds_numpy = np.load(f'{scratch_dir}/{year}_{variation}.npy')
#import matplotlib.pyplot as plt
#plt.hist(ds_numpy[variable_list.index('tasmax')].flatten())
if __name__ == "__main__":
    for year in range(1964,2000):
        for variation in range(1,11):
            print(year,variation)
            convert_to_joined_numpy(year,variation)
        
