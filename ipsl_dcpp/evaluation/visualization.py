import os 
from ipsl_dcpp.model.ipsl_dataset import IPSL_DCPP
import lightning.pytorch as pl
import torch
import hydra
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf,DictConfig
import datetime
import matplotlib.pyplot as plt
import xarray as xr
from celluloid import Camera
import datetime
import subprocess
def inc_time(batch_time):
    batch_time = datetime.datetime.strptime(batch_time,'%Y-%m')
    if(batch_time.month == 12):
        year = batch_time.year + 1
        month = 1
    else:
        year = batch_time.year
        month = batch_time.month + 1
    return f'{year}-{month}'

def gif():
    #gif of rollout
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(16, 6))
    camera = Camera(fig)
    ax1.set_title("predicted")
    ax2.set_title("IPSL_CM6A")
    import xarray as xr
    ds = xr.open_dataset(test.files[0])
    shell = ds.isel(time=0)

    # Animate plot over time
    for time_step in range(118):
        shell[surface_var_name].data = predicted[time_step]
        shell[surface_var_name].plot.pcolormesh(ax=ax1,add_colorbar=False)
        shell[surface_var_name].data = climate_model[time_step]
        shell[surface_var_name].plot.pcolormesh(ax=ax2,add_colorbar=False)
        camera.snap()
    anim = camera.animate()
    anim.save(f"{surface_var_name}_{checkpoint_folder}_rollout.gif")

def rollout(length,dataloader,model):
    iter_dl = iter(dataloader)
    surfaces = []
    plevs = []
    model_plevs = []
    model_surfaces = []
    for i in range(length):
        batch_actual = next(iter_dl)
        if(i == 0):
            batch = batch_actual
        model_surfaces.append(batch_actual['next_state_surface'])
        #model_plevs.append(batch_actual['next_state_level'])
        print(batch['time'])

       # print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
        with torch.no_grad():
            output = model.forward(batch)

       # output['next_state_surface'][:,var_index] = torch.where(land_mask == 1,output['next_state_surface'][:,var_index],0)
        batch=dict(state_surface=output['next_state_surface'],
                   #state_level=output['next_state_level'] + batch['state_level'], 
                   state_depth=output['next_state_depth'],
                   state_constant=batch['state_constant'],
                   next_state_surface=output['next_state_surface'],
                   next_state_depth=output['next_state_depth'],
                  time=[inc_time(batch['time'][0])])
        surfaces.append(batch['state_surface'])
    #    plevs.append(output['next_state_level'])
    return surfaces,model_surfaces

@hydra.main(version_base=None,config_path='../conf',config_name="config.yaml")
def main(cfg: DictConfig):
    scratch = os.environ['SCRATCH']
    work = os.environ['WORK']
    out = subprocess.run('readlink -f wandb/latest-run',shell=True,capture_output=True,text=True)
    run_name = out.stdout.strip('\n').split('-')[-1]
    checkpoint_path = f'{scratch}/checkpoint_{run_name}/epoch=00.ckpt'
    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    test = IPSL_DCPP('test',lead_time_months=1,surface_variables=cfg.experiment.surface_variables,depth_variables=cfg.experiment.depth_variables)
    test_dataloader = torch.utils.data.DataLoader(test,batch_size=1,shuffle=False,num_workers=1)
    model = hydra.utils.instantiate(cfg.experiment.module,backbone=hydra.utils.instantiate(cfg.experiment.backbone),dataset=test_dataloader.dataset)
    model.load_state_dict(checkpoint['state_dict'])
    
    
    surfaces,model_surfaces = rollout(12,test_dataloader,model)
    
    #get shell
    ds = xr.open_dataset(test.files[0])
    shell = ds.isel(time=0)
    var_name = 'gpp'
    #plot lat lon map of first rollout
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 6))
    shell[var_name].data = surfaces[0].squeeze()
    shell[var_name].plot.pcolormesh(ax=ax1)
    plt.savefig(f'images/{run_name}_one_prediction')
    
    #plot rollout timeseries
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 6))
    plt.plot(np.stack(surfaces).squeeze().mean(axis=(1,2)),label='predicted rollout')
    plt.plot(np.stack(model_surfaces).squeeze().mean(axis=(1,2)),label='actual')
    plt.legend()
    plt.title(var_name)
    plt.xlabel('month')
    plt.ylabel('normalized value')
    plt.savefig(f'images/{run_name}_rollout_means')
    #get 12 months of predictions
    trainer = pl.Trainer(fast_dev_run=12)
    output = trainer.predict(model, test_dataloader)
    lat_coeffs_equi = torch.tensor([torch.cos(x) for x in torch.arange(-torch.pi/2, torch.pi/2, torch.pi/143)])
    lat_coeffs =  (lat_coeffs_equi/lat_coeffs_equi.mean())[None, None, None, :, None]
    var_index = cfg.experiment.surface_variables.index(var_name)
    f, axs = plt.subplots(3, 4, figsize=(16, 6))
    axs = axs.flatten()
    #lat_range = 85:100
    #lon_range = 25:50
    subset_lats = lat_coeffs[:,:,:,85:110,:]
    subset_lats = lat_coeffs
    sub_shell = shell.isel(lat=slice(85,110),lon=slice(25,50))
    sub_shell = shell
    for index in range(0,len(output),3):
        pred = output[index][-2]
        batch = output[index][-1]
        predictions = pred['next_state_surface'].squeeze()
        batch_data = batch['next_state_surface'].squeeze()
        predictions = pred['next_state_surface'].squeeze()
        batch_data = batch['next_state_surface'].squeeze()
        #sub_shell[surface_var_name]['time'].data = next(iter(test_dataloader))['time'][0]
        #sub_shell[surface_var_name].data = predictions
        # xx = shell[surface_var_name].plot.pcolormesh(ax=axs[index//3],cmap='ocean',add_colorbar=False,add_labels=False)
        pred_ax = axs[index//3].pcolormesh(predictions,cmap='ocean',vmin=0,vmax=torch.max(batch_data))
        axs[index//3].set_title(f'2014-{index+1}')
        #sub_shell[surface_var_name].data = batch_data
        #shell[surface_var_name].plot.pcolormesh(ax=axs[(index//3) + 4],cmap='ocean',add_colorbar=False,add_labels=False)
        batch_ax = axs[index//3+4].pcolormesh(batch_data ,cmap='ocean',vmin=0,vmax=torch.max(batch_data))


        rmse = torch.sqrt((predictions - batch_data).pow(2).mul(subset_lats))
        #sub_shell[surface_var_name].data = mse.squeeze()
        #shell[surface_var_name].plot.pcolormesh(ax=axs[(index//3) + 8],cmap='ocean',vmax=1e7,vmin=0)
        rmse_ax = axs[(index//3) + 8].pcolormesh(rmse.squeeze(),cmap='ocean',vmin=0)
        axs[(index//3) + 8].set_xlabel('')
        axs[(index//3) + 8].set_title('')
        axs[(index//3) + 8].set_ylabel('')
    f.colorbar(pred_ax, ax=axs[index//3],cmap='ocean',location='right')
    f.colorbar(batch_ax, ax=axs[index//3+4],cmap='ocean',location='right')
    f.colorbar(rmse_ax, ax=axs[index//3+8],cmap='ocean',location='right')
    plt.savefig(f'images/{run_name}_seasonal_predictions_with_errors')
    

if __name__ == "__main__":
    main()
    
    
    
