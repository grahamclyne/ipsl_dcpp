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
train = IPSL_DCPP('train',generate_statistics=True,lead_time_months=1)
train_dataloader = torch.utils.data.DataLoader(IPSL_DCPP('train',0,surface_variables=cfg.experiment.surface_variables,depth_variables=cfg.experiment.depth_variables),batch_size=1,shuffle=False,num_workers=1)
surface_means_out = []
surface_stds_out = []
depth_means_out = []
depth_stds_out = []
iter_batch = iter(train_dataloader)
for count in range(10):
    print(count)
    surface_means = []
    surface_stds = []
    depth_means = []
    depth_stds = []
    for _ in range(120):
        batch = next(iter_batch) 
        surface_means.append(batch['state_surface'].squeeze())
        surface_stds.append(batch['state_surface'].squeeze())
        depth_means.append(batch['state_depth'].squeeze())
        depth_stds.append(batch['state_depth'].squeeze())
    
        

    surface_means_out.append(np.stack(surface_means).reshape(-1,12,91,143,144).mean(axis=0))
    surface_stds_out.append(np.stack(surface_stds).reshape(-1,12,91,143,144).mean(axis=0))
    depth_means_out.append(np.stack(depth_means).reshape(-1,12,3,143,144).mean(axis=0))
    depth_stds_out.append(np.stack(depth_stds).reshape(-1,12,3,143,144).mean(axis=0))
    
np.save('climatology_surface_means.npy',np.stack(surface_means_out).mean(axis=0))
np.save('climatology_surface_stds.npy',np.stack(surface_stds_out).mean(axis=0))
np.save('climatology_depth_means.npy',np.stack(depth_means_out).mean(axis=0))
np.save('climatology_depth_stds.npy',np.stack(depth_stds_out).mean(axis=0))