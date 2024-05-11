import xarray as xr
import os
import torch
from ipsl_dcpp.model.ipsl_dataset import IPSL_DCPP
import hydra
import os
import numpy as np
from hydra import compose, initialize

with initialize(version_base=None, config_path="../conf"):
    cfg = compose(config_name="config")
work_dir = os.environ['WORK']
store_dir = os.environ['STORE'] 

#year = 1960
#variation = 1
#Amon =  xr.open_mfdataset(f'{store_dir}/s{year}-r{variation}i1p1f1/Amon/*.nc',compat='minimal')
#get climatology over train period for a year period
#train = IPSL_DCPP('train',generate_statistics=True,lead_time_months=1)
train =  IPSL_DCPP('train',0,generate_statistics=True,
              delta=False,
              surface_variables=cfg.experiment.surface_variables,
              depth_variables=cfg.experiment.depth_variables,
                  plev_variables=cfg.experiment.plev_variables,
                  normalization='climatology')

train_dataloader = torch.utils.data.DataLoader(
    train,batch_size=1,shuffle=True,num_workers=1)
surface_means_out = []
surface_stds_out = []
depth_means_out = []
depth_stds_out = []
plev_means_out = []
plev_stds_out = []
iter_batch = iter(train_dataloader)
surface_months = [[] for x in range(12)]
depth_months = [[] for x in range(12)]
plev_months = [[] for x in range(12)]

for count in range(1000):
    print(count)
    batch = next(iter_batch) 
    month = int(batch['time'][0].split('-')[-1]) - 1
  #  surface_months[month].append(batch['state_surface'].squeeze())
  #  depth_months[month].append(batch['state_depth'].squeeze())
    plev_months[month].append(batch['state_level'].squeeze())

#     surface_means.append(batch['state_surface'].squeeze())
#     surface_stds.append(batch['state_surface'].squeeze())
#     depth_means.append(batch['state_depth'].squeeze())
#     depth_stds.append(batch['state_depth'].squeeze())

#surface_month_means = [np.nanmean(np.array(x),axis=0) for x in surface_months]
#surface_month_stds = [np.nanstd(np.array(x),axis=(0,2,3)) for x in surface_months]
#depth_month_means = [np.nanmean(np.array(x),axis=0) for x in depth_months]
#depth_month_stds = [np.nanstd(np.array(x),axis=(0,3,4)) for x in depth_months]
plev_month_means = [np.nanmean(np.array(x),axis=0) for x in plev_months]
plev_month_stds = [np.nanstd(np.array(x),axis=(0,3,4)) for x in plev_months]

#np.save('climatology_surface_means.npy',np.stack(surface_month_means))
#np.save('climatology_surface_stds.npy',np.stack(surface_month_stds))
#np.save('climatology_depth_means.npy',np.stack(depth_month_means))
#np.save('climatology_depth_stds.npy',np.stack(depth_month_stds))
np.save('climatology_plev_means.npy',np.stack(plev_month_means))
np.save('climatology_plev_stds.npy',np.stack(plev_month_stds))