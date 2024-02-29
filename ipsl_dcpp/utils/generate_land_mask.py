import xarray as xr
import os
store_dir = os.environ['STORE']
work_dir = os.environ['WORK']
year = 1960
variation = 1
Lmon =  xr.open_mfdataset(f'{store_dir}/s{year}-r{variation}i1p1f1/Lmon/*.nc',compat='minimal')
import numpy as np
land_mask = ~np.isnan(Lmon['gpp'].isel(time=0).data.compute())
np.save(f'{work_dir}/ipsl_dcpp/data/land_mask.npy',land_mask)