from hydra import compose, initialize
import lightning as pl
import hydra
import torch
import glob
import os 
from ipsl_dcpp.utils.visualization_utils import make_gif

# ds = xr.open_dataset('/lustre/fsn1/projects/rech/mlr/udy16au/batch_with_tos/1984_2_tos_included.nc')
# shell = ds.isel(time=4)['tas']
# clim_means = np.load('/lustre/fsn1/projects/rech/mlr/udy16au/reference_data/climatology_surface_means_ensemble_split.npy')
# print(clim_means.shape)
# shell.data = clim_means[11][0]
# shell.plot()

with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config")
device = 'cuda'

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
).to(device)

list_of_files = glob.glob(f'{cfg.exp_dir}/checkpoints/*') 
#list_of_files = glob.glob(f'/gpfsscratch/rech/mlr/udy16au/model_output/ipsl_diffusion/flow_elevation_scaled_250_timesteps/checkpoints/*') 
path = max(list_of_files, key=os.path.getctime)
#path = '/gpfsscratch/rech/mlr/udy16au/model_output/ipsl_diffusion/flow_skip_smaller_embed-p3v4l5/checkpoints/epoch=2-step=4416.ckpt'
checkpoint_path = torch.load(path,map_location=torch.device('cuda'))
pl_module.load_state_dict(checkpoint_path['state_dict'])


rollout_length = 10
batch_timeseries = {'state_surface':[]}
ipsl_ensemble = []
for j in range(0,rollout_length):
    batch = test.__getitem__(j)
    batch_timeseries['state_surface'].append(batch['state_surface'])
    if(j == 0):
        batch = {k:[batch[k]] if k == 'time' or k == 'next_time' else batch[k].unsqueeze(0) for k in batch.keys()}  #simulate lightnings batching dimension
        batch['state_surface'] = batch['state_surface'].to(device)
        batch['prev_state_surface'] = batch['prev_state_surface'].to(device)
        batch['forcings'] = batch['forcings'].to(device)
        batch['solar_forcings'] = batch['solar_forcings'].to(device)#make n for each batch, could do experiments for different num per batch
        batch['state_constant'] = batch['state_constant'].to(device)
     #   rollout = self.sample_rollout(batch,rollout_length=rollout_length,seed = i)
        rollout = pl_module.sample_rollout(batch,rollout_length=rollout_length,seed = j)
        rollout_data = torch.stack(rollout['state_surface'])

ipsl_ensemble = torch.stack(batch_timeseries['state_surface'])
#  batch_timeseries['state_surface'] = torch.where(batch_timeseries['state_surface']==100,0,batch_timeseries['state_surface'])

# ipsl_ensemble.append(batch_timeseries)
# batch_timeseries = {'state_surface':[]}  
# ipsl_ensemble = np.stack(ipsl_ensemble) 
# print(ipsl_ensemble[:,0,:,:].shape)
# print(rollout_data[:,0,0,:,:].shape)
data = torch.stack([ipsl_ensemble[:,0,:,:],rollout_data[:,0,0,:,:].to('cpu')])
print(data.shape)
make_gif(
    data,
    rollout_length,
    'tas',
    0,
    'clim_surface_natural_ckpt',
    True)
