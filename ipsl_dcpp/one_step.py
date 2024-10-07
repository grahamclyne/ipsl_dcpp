import numpy as np
import xarray as xr 
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import lightning as pl
import hydra
import torch
import glob
import os 
from matplotlib.colors import TwoSlopeNorm

work = os.environ['WORK']
print(os.getcwd())
with initialize_config_dir(config_dir=f"{work}/ipsl_dcpp/ipsl_dcpp/conf"):
    cfg = compose(config_name="config")

pl.seed_everything(cfg.seed)
test = hydra.utils.instantiate(
    cfg.dataloader.dataset,domain='test',debug=True
)
test_loader = torch.utils.data.DataLoader(test, 
                                          batch_size=1,
                                          num_workers=cfg.cluster.cpus,
                                          shuffle=False) 
pl_module = hydra.utils.instantiate(
    cfg.module.module,
    backbone=hydra.utils.instantiate(cfg.module.backbone),
    dataset=test_loader.dataset
).to('cuda')

#pl_module.init_from_ckpt('/lustre/fsn1/projects/rech/mlr/udy16au/model_output/ipsl_diffusion/flow_bottom_crop/checkpoints/checkpoint_global_step=55000.ckpt')
list_of_files = glob.glob(f'{cfg.exp_dir}/checkpoints/*') 
#list_of_files = glob.glob(f'/gpfsscratch/rech/mlr/udy16au/model_output/ipsl_diffusion/flow_elevation_scaled_250_timesteps/checkpoints/*') 
path = max(list_of_files)
pl_module.init_from_ckpt(path)
import matplotlib.pyplot as plt
batch = next(iter(test_loader))
for k in batch.keys():
    if(k != 'time' and k != 'next_time'):
        batch[k] = batch[k].to('cuda')
# {k if k == 'time' or k == 'next_time' else batch[k].to('cuda') for k in batch.keys()}  #simulate lightnings batching dimension

rollout_length = 5
rollout = pl_module.sample_rollout(batch,rollout_length=rollout_length,seed = 0)
ds = xr.open_dataset('/lustre/fsn1/projects/rech/mlr/udy16au/batch_with_tos/1961_2_tos_included.nc')
shell = ds.isel(time=4)['tas']

# print(rollout['state_surface'][0][0][0].shape)
# print(rollout['state_surface'][0][0][0])
vmin = torch.min(rollout['state_surface'][0][0][0]).cpu()
vmax = torch.max(rollout['state_surface'][0][0][0]).cpu()
norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmin+vmax)/2,vmax=vmax)

fig,ax = plt.subplots(rollout_length+1,figsize=(6,12))
ax = ax.flatten()
ax[0].pcolormesh(rollout['state_surface'][0][0][0][:143,:].cpu(),norm=norm)
for rollout_index in range(rollout_length): 
    shell.data = rollout['next_state_surface'][rollout_index][0][0][:143,:].cpu()
    ax[rollout_index+1].pcolormesh(shell.data,norm=norm)

fig.savefig('one_step.png')