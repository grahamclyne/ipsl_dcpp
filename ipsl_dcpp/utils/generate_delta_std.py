import os
import torch
import hydra
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
    cfg.dataloader.dataset,
    domain='train',
    generate_statistics=False, #want normalized values
    delta=False,
)


surface_deltas = []
depth_deltas = []
plev_deltas = []

for count in range(1000):
    print(count)
    sample_idx = torch.randint(len(train), size=(1,)).item()
    batch = train[sample_idx]
    surface_delta = (batch['next_state_surface'] - batch['state_surface']).squeeze()
   # depth_delta = (batch['next_state_depth'] - batch['state_depth']).squeeze()
    plev_delta = (batch['next_state_level'] - batch['state_level']).squeeze()
   # depth_deltas.append(depth_delta)
    surface_deltas.append(surface_delta)
    plev_deltas.append(plev_delta)

np.save('surface_delta_std_ensemble_split',np.nanstd(np.stack(surface_deltas),axis=0))
#np.save('depth_delta_std',np.nanstd(np.stack(depth_deltas),axis=0))
np.save('plev_delta_std_ensemble_split',np.nanstd(np.stack(plev_deltas),axis=0))
