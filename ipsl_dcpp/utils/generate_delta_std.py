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
    delta=False,
    plev_variables=cfg.experiment.plev_variables,
    work_path=cfg.environment.work_path,
    scratch_path=cfg.environment.scratch_path,
)
train_dataloader = torch.utils.data.DataLoader(
    train,
    batch_size=1,
    shuffle=True,
    num_workers=1,
)

iter_batch = iter(train_dataloader)

surface_deltas = []
depth_deltas = []
plev_deltas = []

for count in range(1000):
    print(count)
    batch = next(iter_batch)
   # surface_delta = (batch['next_state_surface'] - batch['state_surface']).squeeze()
   # depth_delta = (batch['next_state_depth'] - batch['state_depth']).squeeze()
    plev_delta = (batch['next_state_level'] - batch['state_level']).squeeze()
   # depth_deltas.append(depth_delta)
   # surface_deltas.append(surface_delta)
    plev_deltas.append(plev_delta)

#np.save('surface_delta_std',np.nanstd(np.stack(surface_deltas),axis=0))
#np.save('depth_delta_std',np.nanstd(np.stack(depth_deltas),axis=0))
np.save('plev_delta_std',np.nanstd(np.stack(plev_deltas),axis=0))
