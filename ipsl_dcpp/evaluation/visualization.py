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
import sys
import scipy.stats as stats
import xarray as xr

    #this refers to the amount of energy found per unit of time - assumes to be over all of time!!
    #average power of a sinusoid is amplitude^2 / 2
    
    #take fourier transform put data into spectral dimension, take average of power spectra of all the powers 
def get_power_spectral_density(data):
    #data = data[:data.shape[0], data.shape[1]//2-data.shape[0]//2:data.shape[1]//2+data.shape[0]//2]
    #have to make this square, clip the last row of lats........
    data = data[:,:data.shape[0]]
    npix = data.shape[0]
    #print(npix)
    #print(data.shape)
    #take the fourier image - i.e. the spectral data in the frequency domain 
    fourier_image = np.fft.fftn(data)
    #take the amplitudes, and one side of the fourier image (it is symmetrical)
    #why take the square here? 
    
    #we can find the amplitude of a sinusoidal by taking the fft and scaling it 
    #look at formula to find power from a coefficient - take abs and square https://en.wikipedia.org/wiki/Spectral_density
    fourier_amplitudes = np.abs(fourier_image)**2
    # power is just the 

    #now return Discrete Fourier Transform sample frequencies from npix... what? 
    kfreq = np.fft.fftfreq(npix) * npix # this gives the freq in hertz, need to mulptiply by "frame rate" i.e. number of pixels? 
    kfreq2D = np.meshgrid(kfreq, kfreq)
  #  print(kfreq2D[0].shape)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1]) #center each bin?
  #  print(fourier_amplitudes.shape)
  #  print(knrm.shape)
  #  print(kbins)
    print(knrm)
    Abins, _, _ = stats.binned_statistic(knrm, 
                                         fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    return kvals, Abins


def inc_time(batch_time):
    batch_time = datetime.datetime.strptime(batch_time,'%Y-%m')
    if(batch_time.month == 12):
        year = batch_time.year + 1
        month = 1
    else:
        year = batch_time.year
        month = batch_time.month + 1
    return f'{year}-{month}'

def gif(predicted,actual,surface_var_name,test_dataset,length):
    #gif of rollout
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(16, 6))
    camera = Camera(fig)
    ax1.set_title("predicted")
    ax2.set_title("IPSL_CM6A")
    ds = xr.open_dataset(test_dataset.files[0])
    shell = ds.isel(time=0)

    # Animate plot over time
    for time_step in range(length):
        shell[surface_var_name].data = predicted[time_step]
        shell[surface_var_name].plot.pcolormesh(ax=ax1,add_colorbar=False)
        shell[surface_var_name].data = actual[time_step]
        shell[surface_var_name].plot.pcolormesh(ax=ax2,add_colorbar=False)
        camera.snap()
    anim = camera.animate()
    anim.save(f"{surface_var_name}_rollout.gif")

    
    

def rollout(length,dataloader,model,zeroes):
    iter_dl = iter(dataloader)
    surfaces = []
    plevs = []
    model_plevs = []
    model_surfaces = []
    for i in range(length):
        batch_actual = next(iter_dl)
        print(batch_actual['time'])
        #print(batch_actual['next_state_surface'].shape)
        #print(np.expand_dims(dataloader.dataset.surface_delta_stds,0).shape)
        if(dataloader.dataset.delta == True):
            batch_actual['next_state_surface'] = (batch_actual['next_state_surface']*np.expand_dims(dataloader.dataset.surface_delta_stds,0)) + batch_actual['state_surface']

        #_,batch_actual = dataloader.dataset.denormalize(None,batch_actual)
        if(i == 0):
            batch = batch_actual
            surfaces.append(batch)
            model_surfaces.append(batch)
           # if(dataloader.dataset.delta == True):
           #     batch_actual['next_state_surface'] = batch_actual['next_state_surface']*np.expand_dims(dataloader.dataset.surface_delta_stds,0) + batch_actual['state_surface']
        else:
            model_surfaces.append(batch_actual)
        #model_plevs.append(batch_actual['next_state_level'])
      #  batch['state_surface'] = batch['state_surface'].unsqueeze(0)
       # batch['state_depth'] = batch['state_depth'].unsqueeze(0)

       # print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
        if(zeroes == True):
            batch['state_surface'] = torch.zeros(batch['state_surface'].shape)
            batch['state_depth'] = torch.zeros(batch['state_depth'].shape)

        with torch.no_grad():
            output = model.forward(batch)
        #land_mask = batch['state_constant']
     #   output['next_state_surface'][:,89] = torch.where(land_mask.squeeze() == 1,output['next_state_surface'][:,89],0)
        if(dataloader.dataset.delta == True):
          #  print(np.expand_dims(dataloader.dataset.surface_delta_stds,0).shape)
          #  print(output['next_state_surface'].unsqueeze(0).shape)
          #  print(batch['state_surface'].shape)
            batch=dict(
                state_surface=output['next_state_surface'].unsqueeze(0)*np.expand_dims(dataloader.dataset.surface_delta_stds,0) + batch['state_surface'],
                   #state_level=output['next_state_level'] + batch['state_level'], 
                state_depth=output['next_state_depth'].unsqueeze(0)*np.expand_dims(dataloader.dataset.depth_delta_stds,0) + batch['state_depth'],
                state_constant=batch['state_constant'],
                next_state_surface= output['next_state_surface'].unsqueeze(0)*np.expand_dims(dataloader.dataset.surface_delta_stds,0) + batch['state_surface'],
                next_state_depth=output['next_state_depth'].unsqueeze(0)*np.expand_dims(dataloader.dataset.depth_delta_stds,0) + batch['state_depth'],
                time=[inc_time(batch['time'][0])])
        else:
            batch=dict(state_surface=output['next_state_surface'].unsqueeze(0),
                   #state_level=output['next_state_level'] + batch['state_level'], 
                   state_depth=output['next_state_depth'].unsqueeze(0),
                   state_constant=batch['state_constant'],
                   next_state_surface= output['next_state_surface'].unsqueeze(0),
                   next_state_depth=output['next_state_depth'].unsqueeze(0),
                  time=[inc_time(batch['time'][0])])                                  
        surfaces.append(batch)
    #    plevs.append(output['next_state_level'])
    return surfaces,model_surfaces

@hydra.main(version_base=None,config_path='../conf',config_name="config.yaml")
def main(cfg: DictConfig):
    scratch = os.environ['SCRATCH']
    work = os.environ['WORK']
    
    run_id = ''
    out = subprocess.run(f'ls -tr {scratch}/checkpoint_{run_id}/ | tail -n 1',shell=True,capture_output=True,text=True)
    path = out.stdout.strip("\n")
    print(path)
    checkpoint_path = f'{scratch}/checkpoint_{run_id}/{path}'
    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    test = IPSL_DCPP('test',lead_time_months=1,surface_variables=cfg.experiment.surface_variables,depth_variables=cfg.experiment.depth_variables)
    test_dataloader = torch.utils.data.DataLoader(test,batch_size=1,shuffle=False,num_workers=1)
    model = hydra.utils.instantiate(cfg.experiment.module,backbone=hydra.utils.instantiate(cfg.experiment.backbone),dataset=test_dataloader.dataset)
    model.load_state_dict(checkpoint['state_dict'])
    
    rollout_steps = 12
    surfaces,model_surfaces = rollout(rollout_steps,test_dataloader,model)
    
    #get shell
    ds = xr.open_dataset(test.files[0])
    shell = ds.isel(time=0)
    var_name = 'gpp'
    var_index = cfg.experiment.surface_variables.index(var_name)

    #plot lat lon map of first rollout
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 6))
    shell[var_name].data = surfaces[0].squeeze()[var_index]
    shell[var_name].plot.pcolormesh(ax=ax1)
  #  plt.savefig(f'images/{run_id}_one_prediction')
    #plot rollout timeseries
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 6))
    
    stacked_surface_variable = np.stack(surfaces).squeeze()[:,var_index]
    plt.plot(stacked_surface_variable.mean(axis=(-1,-2)),label='predicted rollout')
    ci = 1.96 * stacked_surface_variable.std(axis=(-1,-2))/np.sqrt(len(stacked_surface_variable))
    ax1.fill_between(stacked_surface_variable.std(axis=(-1,-2)), (stacked_surface_variable.mean(axis=(-1,-2))-ci), (stacked_surface_variable.mean(axis=(-1,-2))+ci), color='b', alpha=.1)
    plt.plot(np.stack(model_surfaces).squeeze()[:,var_index].mean(axis=(-1,-2)),label='actual')
    #ax.fill_between(x, (y-ci), (y+ci), color='b', alpha=.1)

    plt.legend()
    plt.title(var_name)
    plt.xlabel('month')
    plt.ylabel('normalized value')
    plt.savefig(f'images/{run_id}_{var_name}_rollout_means')
    
    
    #plot power spectra
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=300)
    fig.suptitle('Power spectral density')
    freqs_1 = []
    amps_1 = []
    freqs_2 = []
    amps_2 = []
    
    for step in range(rollout_steps):
        logfreq1, logamp1 = get_power_spectral_density(surfaces[step][:,var_index].squeeze())
        logfreq2, logamp2 = get_power_spectral_density(model_surfaces[step][:,var_index].squeeze())
        freqs_1.append(logfreq1)
        amps_1.append(logamp1)
        freqs_2.append(logfreq2)
        amps_2.append(logamp2)

    ax.loglog(np.stack(freqs_1).mean(axis=0),np.stack(amps_1).mean(axis=0), label='pangu')
    ax.loglog(np.stack(freqs_2).mean(axis=0),np.stack(amps_2).mean(axis=0), label='ipsl')

    #axes.loglog(logfreq2, logamp2, label='output')
    ax.grid()
    ax.legend()
    ax.set_title(var_name)
    
    plt.savefig(f'images/{run_id}_{var_name}_power_spectra')
    
    
    
    
    
    #get 12 months of predictions
    trainer = pl.Trainer(fast_dev_run=12)
    output = trainer.predict(model, test_dataloader)
    lat_coeffs_equi = torch.tensor([torch.cos(x) for x in torch.arange(-torch.pi/2, torch.pi/2, torch.pi/143)])
    lat_coeffs =  (lat_coeffs_equi/lat_coeffs_equi.mean())[None, None, None, :, None]
    f, axs = plt.subplots(3, 4, figsize=(16, 6))
    axs = axs.flatten()

    subset_lats = lat_coeffs[:,:,:,85:110,:]
    subset_lats = lat_coeffs
    sub_shell = shell.isel(lat=slice(85,110),lon=slice(25,50))
    sub_shell = shell
    for index in range(0,len(output),3):
        pred = output[index][-2]
        batch = output[index][-1]
        pred,batch = test.denormalize(pred,batch)
        print(pred['next_state_surface'].shape)
        predictions = pred['next_state_surface'].squeeze()[var_index]
        batch_data = batch['next_state_surface'].squeeze()[var_index]
     #   predictions = pred['next_state_surface'].squeeze()[:,var_index]
     #   batch_data = batch['next_state_surface'].squeeze()[:,var_index]
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
    plt.savefig(f'images/{run_id}_{var_name}_seasonal_predictions_with_errors')
    

if __name__ == "__main__":
    main()
    
    
    
