import xesmf as xe
import xarray as xr

#xesmf 0.7.0
#https://github.com/JiaweiZhuang/xESMF/issues/78
import os 
scratch = os.environ['SCRATCH']
store = os.environ['STORE']
regridder = None
for year in range(1999,2000):
    for ensemble in range(1,11):
        print(year,ensemble)
        test = xr.open_dataset(f'{scratch}/{year}_{ensemble}.nc')
        #omon = xr.open_dataset(f'{scratch}/omon_reproj/omon_reproj_{year}_{ensemble}.nc')
        if 'tos' in list(test.keys()):
            test = test.drop_vars(['tos'])
        if 'x' in list(test.keys()) or 'x' in list(test.coords):
            test = test.drop_vars(['x'])
        if 'y' in list(test.keys()) or 'y' in list(test.coords):
            test = test.drop_vars(['y'])
        if 'nav_lon' in list(test.keys()) or 'nav_lon' in list(test.coords):
            test = test.drop_vars(['nav_lon'])
        if 'nav_lat' in list(test.keys()) or 'nav_lat' in list(test.coords):
            test = test.drop_vars(['nav_lat'])
        omon =  xr.open_mfdataset(f'{store}/s{year}-r{ensemble}i1p1f1/Omon/*.nc')
       # test = test.drop_vars(['x','y'])
        omon = omon.rename({'y':'lat','x':'lon'})
        omon = omon.isel(lon=slice(1,None)).compute()
      #  omon = omon.drop_vars(['nav_lon','nav_lat'])
        # print(omon)
        # print(test)
        if(not regridder):
            regridder = xe.Regridder(omon, test, "conservative",ignore_degenerate=True)
        omon_out = regridder(omon)
        omon_out = omon_out.drop_vars(['time','lat','lon','area'])
        omon_out['tos'].isel(time=0).plot()

            
        #omon = omon.drop_vars(['x','y'])
        #omon = omon.rename({'x':'lon','y':'lat'})
        merged = xr.merge([omon_out,test])
        merged.to_netcdf(f'{scratch}/{year}_{ensemble}_tos_included.nc')
        test.close()
        merged.close()
        omon.close()
