import xarray as xr 
import xesmf as xe
import os 

ds = xr.open_dataset('ETOPO_2022_v1_60s_N90W180_bed.nc')
scratch = os.environ['SCRATCH']
ds1 = xr.open_dataset(f'{scratch}/batch_with_tos/1994_4_tos_included.nc')

# print(test)
regridder = xe.Regridder(ds1, ds, "conservative",ignore_degenerate=True)
omon_out = regridder(ds1)
# omon_out = omon_out.drop_vars(['time','lat','lon','area'])
# omon_out['tos'].isel(time=0).plot()
omon_out.to_netcdf('regridded_elevation.nc')
