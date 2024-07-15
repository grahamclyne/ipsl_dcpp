#from ipsl_dcpp.model.ipsl_dataset import IPSL_DCPP
import torch
import lightning as pl
#from ipsl_dcpp.model.pangu import PanguWeather
from hydra import compose, initialize
from omegaconf import OmegaConf
import numpy as np
import hydra
import os
import pickle
import io
import numpy as np
from matplotlib import animation
import xarray as xr 
#os.environ['SLURM_NTASKS_PER_NODE'] = '1'
#torch.set_default_dtype(torch.float32)
# os.environ["CUDA_VISIBLE_DEVICES"]=""
#torch.set_default_tensor_type(torch.FloatTensor)

with initialize(version_base=None, config_path="../conf"):
    cfg = compose(config_name="config")
pl.seed_everything(cfg.seed)
val = hydra.utils.instantiate(
    cfg.dataloader.dataset,domain='val'
)
val_loader = torch.utils.data.DataLoader(val, 
                                            batch_size=1,
                                            num_workers=0,
                                            shuffle=True) 

ensembles = []
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
num_ensembles = 10
for i in range(0,num_ensembles):
   # ensembles.append(CPU_Unpickler(open(f'../{i}_rollout_v_predictions_30year_ssp_585_20e12882_50steps_fixed_velocity.pkl', 'rb')).load())
    ensembles.append(CPU_Unpickler(open(f'{cfg.exp_dir}/rollouts/{i}_rollout_{cfg.name}.pkl', 'rb')).load())

num_steps = len(ensembles[0]['state_surface'])
# fixed = CPU_Unpickler(open(f'../0_rollout_v_predictions_30year_ssp_585_20e12882_50steps_fixed.pkl', 'rb')).load()

#LOAD BATCH DATA
ipsl_ensemble = []
batch_timeseries = {'state_surface':[]}
# batch_iter = iter(val_loader)
# num_members = 3
for i in range(0,num_ensembles):
    for j in range(0,118):
        batch = val.__getitem__(i*118 + j)
        batch_timeseries['state_surface'].append(batch['state_surface'])
    batch_timeseries['state_surface'] = np.stack(batch_timeseries['state_surface'])
    ipsl_ensemble.append(batch_timeseries)
    batch_timeseries = {'state_surface':[]}    
# for b in range((118 * num_members) + 1):
#     if(b % 118 == 0 and b != 0):
#         batch_timeseries['state_surface'] = np.stack(batch_timeseries['state_surface'])
#         ipsl_ensemble.append(batch_timeseries)
#         batch_timeseries = {'state_surface':[]}
#     #batch = next(batch_iter)
#     print(batch['time'])
#     batch_timeseries['state_surface'].append(batch['state_surface'])
ipsl_ensemble = np.stack(ipsl_ensemble) 

#verify batches 
import matplotlib.pyplot as plt
# file_name = f'{cfg.exp_dir}/rollouts/{i}_rollout_{cfg.name}.pkl'
# output_file = Path("/foo/bar/baz.txt")
# output_file.parent.mkdir(exist_ok=True, parents=True)
# file_name = f'{cfg.exp_dir}/plots/ensemble_verificatoin.png'
# for i in ipsl_ensemble:
#     plt.plot(np.nanmean(i['state_surface'][:,0,0],axis=(-1,-2))).savefig()

batch_denormed = []
denormed_surface_ensembles = []
denormed_batch_surface_ensembles = []
denormed_plev_ensembles = []
denormed_batch_plev_ensembles = []
for i in range(num_ensembles):
    month_index = 0
    denormalized_surface = []
    denormalized_plev = []
    batch_denormed_surface = []
    batch_denormed_plev = []

    for index in range(num_steps-6):
   # for index in range(len(ensembles[i]['state_surface'])-2):
        denorm_surface = lambda x,month_index: x[:10]*torch.from_numpy(val.surface_stds[month_index]) + torch.from_numpy(val.surface_means[month_index])
        denorm_plev = lambda x,month_index: x[10:].reshape(-1,8,3,143,144)*torch.from_numpy(val.plev_stds[month_index]) + torch.from_numpy(val.plev_means[month_index])
       # if(cfg.dataloader.flattened_plev):
#        print(ensembles[i]['state_surface'][index].shape)
 #       print(ipsl_ensemble[i]['state_surface'][index].shape)
        if(val.flattened_plev):
            denormalized_plev.append(denorm_plev(ensembles[i]['state_surface'][index][0],month_index))
            batch_denormed_plev.append(denorm_plev(torch.Tensor(ipsl_ensemble[i]['state_surface'][index]),month_index))

        denormalized_surface.append(denorm_surface(ensembles[i]['state_surface'][index][0],month_index))
        batch_denormed_surface.append(denorm_surface(torch.Tensor(ipsl_ensemble[i]['state_surface'][index]),month_index))
        if(month_index == 11):
            month_index = 0
        else:
            month_index += 1
    denormed_surface_ensembles.append(denormalized_surface)
    denormed_batch_surface_ensembles.append(batch_denormed_surface)
    if(val.flattened_plev):
        denormed_plev_ensembles.append(denormalized_plev)
        denormed_batch_plev_ensembles.append(batch_denormed_plev)
out_dir = f'{cfg.exp_dir}/plots'

num_vars = 10
minimum = 1000
maximum = -1000
for var_num in range(0,num_vars):
    fig, axes = plt.subplots(2, figsize=(16, 6))
    axes = axes.flatten()
    for i in ensembles:
        axes[0].plot(np.mean(np.stack(i['state_surface'])[:num_steps,0,var_num],axis=(-1,-2)))
    # min = np.mean(np.stack(i['state_surface'])[:num_steps,0,var_num],axis=(-1,-2)).min()
    # max = np.mean(np.stack(i['state_surface'])[:num_steps,0,var_num],axis=(-1,-2)).max()
        minimum = minimum if minimum < np.mean(np.stack(i['state_surface'])[:num_steps,0,var_num],axis=(-1,-2)).min() else np.mean(np.stack(i['state_surface'])[:num_steps,0,var_num],axis=(-1,-2)).min()
        maximum = maximum if maximum > np.mean(np.stack(i['state_surface'])[:num_steps,0,var_num],axis=(-1,-2)).max() else np.mean(np.stack(i['state_surface'])[:num_steps,0,var_num],axis=(-1,-2)).max()
    for i in np.stack(ipsl_ensemble):
        axes[1].plot(np.nanmean(i['state_surface'][:num_steps,var_num],axis=(-1,-2)))
        minimum = minimum if minimum < np.mean(np.stack(i['state_surface'])[:num_steps,var_num],axis=(-1,-2)).min() else np.mean(np.stack(i['state_surface'])[:num_steps,var_num],axis=(-1,-2)).min()
        maximum =maximum if maximum > np.mean(np.stack(i['state_surface'])[:num_steps,var_num],axis=(-1,-2)).max() else np.mean(np.stack(i['state_surface'])[:num_steps,var_num],axis=(-1,-2)).max()
    axes[0].set_ylim(minimum,maximum)
    axes[1].set_ylim(minimum,maximum)
    axes[0].set_title('Predicted')
    axes[1].set_title('IPSL')
    axes[0].set_ylabel('Normalized Value')
    axes[1].set_ylabel('Normalized Value')
    file_name = f'{out_dir}/normalized_comparison_var_{var_num}.png'
    from pathlib import Path
    output_file = Path(file_name)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    plt.title(cfg.module.surface_variables[var_num])
    fig.savefig(file_name)


    ds = xr.open_dataset(val.files[0])
    shell = ds.isel(time=0)
    fig, axes = plt.subplots(1,2, figsize=(16, 6))
    axes = axes.flatten()
    container = []
    for time_step in range(num_steps-6):
        # print(np.stack(ensembles[0]['state_surface']).shape)
        # print(np.stack(ipsl_ensemble[0]['state_surface']).shape)
        shell['tas'].data = np.stack(ensembles[0]['state_surface'])[time_step][0][var_num]
       # line = ax1.pcolormesh(steps[time_step][0,0,0])
        line = shell['tas'].plot.pcolormesh(ax=axes[0],add_colorbar=False)
        shell['tas'].data = np.stack(ipsl_ensemble[0]['state_surface'])[time_step][var_num]
        line1 = shell['tas'].plot.pcolormesh(ax=axes[1],add_colorbar=False)
        title = axes[0].text(0.5,1.05,"Diffusion Step {}".format(time_step), 
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=axes[0].transAxes,)
        axes[0].set_title('Predicted')
        axes[1].set_title('IPSL')
    
        container.append([line, line1,title])
    plt.title(cfg.module.surface_variables[var_num])
    
    ani = animation.ArtistAnimation(fig, container, interval=200, blit=True)
    ani.save(f'{out_dir}/diffusion_comparison_{var_num}.gif')


#DO PLEV NOW
if(val.flattened_plev):
    num_vars = 10
    for var_num in range(10,num_vars+10):
        fig, axes = plt.subplots(1,2, figsize=(16, 6))
        axes = axes.flatten()
        container = []
        for time_step in range(num_steps-6):
            shell['tas'].data = np.stack(ensembles[0]['state_surface'])[time_step][0][var_num]
           # line = ax1.pcolormesh(steps[time_step][0,0,0])
            line = shell['tas'].plot.pcolormesh(ax=axes[0],add_colorbar=False)
            shell['tas'].data = np.stack(ipsl_ensemble[0]['state_surface'])[time_step][var_num]
            line1 = shell['tas'].plot.pcolormesh(ax=axes[1],add_colorbar=False)
            title = axes[0].text(0.5,1.05,"Diffusion Step {}".format(time_step), 
                            size=plt.rcParams["axes.titlesize"],
                            ha="center", transform=axes[0].transAxes,)
            axes[0].set_title('Predicted')
            axes[1].set_title('IPSL')
        
            container.append([line, line1,title])
        plt.title(cfg.module.surface_variables[var_num])
        
        ani = animation.ArtistAnimation(fig, container, interval=200, blit=True)
        ani.save(f'{out_dir}/diffusion_comparison_plev_{var_num}.gif')
