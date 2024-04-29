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
train = hydra.utils.instantiate(
    cfg.experiment.train_dataset,
    generate_statistics=False,
    surface_variables=cfg.experiment.surface_variables,
    depth_variables=cfg.experiment.depth_variables,
    delta=False
)
train_dataloader = torch.utils.data.DataLoader(
    train,
    batch_size=1,
    shuffle=True,
    num_workers=1,
)
surface_means_out = []
surface_stds_out = []
depth_means_out = []
depth_stds_out = []
iter_batch = iter(train_dataloader)

surface_deltas = []
depth_deltas = []
for count in range(1000):
    print(count)
    batch = next(iter_batch)
    surface_delta = (batch['next_state_surface'] - batch['state_surface']).squeeze()
    depth_delta = (batch['next_state_depth'] - batch['state_depth']).squeeze()
    print(np.nanmean(surface_delta))
    depth_deltas.append(depth_delta)
    surface_deltas.append(surface_delta)
    
np.save('surface_delta_std',np.nanstd(np.stack(surface_deltas),axis=0))
np.save('depth_delta_std',np.nanstd(np.stack(depth_deltas),axis=0))
