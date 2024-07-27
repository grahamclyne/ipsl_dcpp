import lightning.pytorch as pl
import diffusers
import torch
import wandb
from ipsl_dcpp.model.pangu import TimestepEmbedder
from ipsl_dcpp.evaluation.metrics import EnsembleMetrics
import datetime
import numpy as np
import time
import pandas as pd
import os 
import pickle
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib import animation
from copy import deepcopy


def visualize_denoise(steps,dataset):
    #visualize denoising proccess
    from matplotlib import animation
    import xarray as xr
    import matplotlib.pyplot as plt
    ds = xr.open_dataset(dataset.files[0])
    shell = ds.isel(time=0)
    fig, axes = plt.subplots(1,1, figsize=(16, 6))
    steps = np.stack([x.cpu() for x in steps])
    container = []
    var_num = -1
    for time_step in range(len(steps)):
        # print(np.stack(ensembles[0]['state_surface']).shape)
        # print(np.stack(ipsl_ensemble[0]['state_surface']).shape)
        shell['tas'].data = steps[time_step][0][var_num]
       # line = ax1.pcolormesh(steps[time_step][0,0,0])
        line = shell['tas'].plot.pcolormesh(ax=axes,add_colorbar=False,vmax=5,vmin=-5)
        title = axes.text(0.5,1.05,"Diffusion Step {}".format(time_step), 
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=axes.transAxes,)
        axes.set_title('denoise')
    
        container.append([line,title])
    plt.title('')
    
    ani = animation.ArtistAnimation(fig, container, interval=200, blit=True)
    ani.save(f'denoise_{var_num}_epsilon.gif')


   #  fig, axes = plt.subplots(1,1, figsize=(16, 6))
   #  var_num = -1
   #  shell['tas'].data = sample['next_state_surface'][0][var_num].cpu()
   # # line = ax1.pcolormesh(steps[time_step][0,0,0])
   #  line = shell['tas'].plot.pcolormesh(ax=axes,add_colorbar=True,vmax=5,vmin=-5)
   #  fig.savefig(f'denoised_image_epsilon__{var_num}.png')


def inc_time(batch_time):
    batch_time = datetime.datetime.strptime(batch_time,'%Y-%m')
    if(batch_time.month == 12):
        year = batch_time.year + 1
        month = 1
    else:
        year = batch_time.year
        month = batch_time.month + 1
    return f'{year}-{month}'


pressure_levels = torch.tensor([  50,  100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925,
       1000]).float()
surface_coeffs = torch.tensor([0.1, 0.1, 1.0, 0.1])[None, :, None, None, None] # graphcast
surface_coeffs = torch.tensor([0.25, 0.25, 0.25, 0.25])[None, :, None, None, None] # pangu coeffs
level_coeffs = (pressure_levels/pressure_levels.mean())[None, None, :, None, None]

def bw_to_bwr(bw_tensor, m=None, M=None):
    x = bw_tensor
    if m is None:
        x = x - x.min()
        x = x / x.max()
    else:
        x = (x - m) / (M - m)
    red = torch.tensor([1, 0, 0])[:, None, None].to(x.device)
    white = torch.tensor([1, 1, 1])[:, None, None].to(x.device)
    blue = torch.tensor([0, 0, 1])[:, None, None].to(x.device)
    x_blue = blue + 2*x*(white-blue)
    x_red = white + (2*x - 1)*(red-white)
    x_bwr = x_blue * (x < 0.5) + x_red * (x >= 0.5)
    x_bwr = (x_bwr * 255).int().permute((1, 2, 0))
    x_bwr = x_bwr.cpu().numpy().astype('uint8')
    return x_bwr

class Diffusion(pl.LightningModule):
    def __init__(
        self,
        num_diffusion_timesteps,
        backbone,
        lr,
        betas,
        weight_decay,
        num_warmup_steps,
        num_training_steps,
        num_cycles,
        dataset,
        num_inference_steps,
        prediction_type,
        num_ensemble_members,
        p_uncond,
        num_rollout_steps,
        scheduler
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.month_embedder = TimestepEmbedder(256)
        self.timestep_embedder = TimestepEmbedder(256)
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.backbone = backbone # necessary to put it on device
        self.dataset = dataset
        self.metrics = EnsembleMetrics(dataset=dataset)
        self.num_inference_steps = num_inference_steps
        self.scheduler = scheduler
        self.num_members = num_ensemble_members
        self.num_rollout_steps = num_rollout_steps
        if scheduler == 'ddpm':
            self.noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=num_diffusion_timesteps,
                                                           beta_schedule='squaredcos_cap_v2',
                                                           beta_start=0.0001,
                                                           beta_end=0.012,
                                                           prediction_type=prediction_type,
                                                           clip_sample=False,
                                                           clip_sample_range=1e6,
                                                           rescale_betas_zero_snr=True,
                                                           )
        elif scheduler == 'flow':
            from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=num_diffusion_timesteps)


        self.p_uncond = p_uncond


    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
       
    def mylog(self, dct={}, **kwargs):
        mode = 'train_' if self.training else 'val_'
        dct.update(kwargs)
        for k, v in dct.items():
            self.log(mode+k, v, prog_bar=True,sync_dist=True)

    def forward(self, batch, timesteps, sel=1):
        bs = batch['state_surface'].shape[0]
        device = batch['state_surface'].device
        batch['state_surface'] = torch.cat([batch['state_surface']*sel,batch['prev_state_surface']*sel, 
                                   batch['surface_noisy']], dim=1)
        if(self.backbone.plev):
            batch['state_level'] = torch.cat([batch['state_level']*sel,batch['prev_state_level']*sel, 
                                   batch['level_noisy']], dim=1)
        month = torch.tensor([int(x[5:7]) for x in batch['time']]).to(device)
        year = torch.tensor([int(x[0:4]) for x in batch['time']]).to(device)
        month_emb = self.month_embedder(month)
        year_emb = self.month_embedder(year)
        timestep_emb = self.timestep_embedder(timesteps)
        ch4_emb = self.timestep_embedder(batch['forcings'][:,0])
        cfc11_emb = self.timestep_embedder(batch['forcings'][:,1])
        cfc12_emb = self.timestep_embedder(batch['forcings'][:,2])
        c02_emb = self.timestep_embedder(batch['forcings'][:,3])

        cond_emb = (month_emb + year_emb + timestep_emb + ch4_emb + cfc11_emb + cfc12_emb + c02_emb)
        for wavelength_index in range(len(batch['solar_forcings'][0])):
            cond_emb += self.timestep_embedder(batch['solar_forcings'][:,wavelength_index]) 
        out = self.backbone(batch, cond_emb)
        return out


    
    def training_step(self, batch, batch_nb):
        device = batch['state_surface'].device
        bs = batch['state_surface'].shape[0]

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
        surface_noise = torch.randn_like(batch['next_state_surface'])


        if self.scheduler == 'flow':
            #batch['surface_noisy'] = self.noise_scheduler.scale_noise(batch['next_state_surface'], timesteps, surface_noise) # sample timesteps noise
            #batch['level_noisy'] = self.noise_scheduler.scale_noise(batch['next_state_level'], timesteps, level_noise)
            #target_surface = batch['next_state_surface'] - surface_noise
            #target_level = batch['next_state_level'] - level_noise
            
            # weighting scheme: logit normal
            u = torch.normal(mean=0, std=1, size=(bs,), device='cpu').sigmoid()
            
            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler.timesteps[indices].to(device=device)
            
            def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
                sigmas = self.noise_scheduler.sigmas.to(device=device, dtype=dtype)
                schedule_timesteps = self.noise_scheduler.timesteps.to(device)
                timesteps = timesteps.to(device)
                step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
            
                sigma = sigmas[step_indices].flatten()
                while len(sigma.shape) < n_dim:
                    sigma = sigma.unsqueeze(-1)
                return sigma
                    
            # Add noise according to flow matching.
            sigmas = get_sigmas(timesteps, n_dim=batch['next_state_surface'].ndim, dtype=batch['next_state_surface'].dtype)
            
            batch['surface_noisy'] = noisy_surface_input = sigmas * surface_noise + (1.0 - sigmas) * batch['next_state_surface']
          #  batch['level_noisy'] = noisy_level_input = sigmas * level_noise + (1.0 - sigmas) * batch['next_state_level']

            target_surface = surface_noise - batch['next_state_surface']
         #   target_level = level_noise - batch['next_state_level']
        else:             
            batch['surface_noisy'] = self.noise_scheduler.add_noise(batch['next_state_surface'], surface_noise, timesteps)
            batch['surface_noisy'] = self.noise_scheduler.scale_model_input(batch['surface_noisy'])
            if(self.backbone.plev):
                level_noise = troch.randn_like(batch['next_state_level'])
                batch['level_noisy'] = self.noise_scheduler.add_noise(batch['level_noisy'], level_noise,timesteps)
                batch['level_noisy'] = self.noise_scheduler.scale_model_input(batch['level_noisy'])
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target_surface = surface_noise
               # target_level = level_noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target_surface = self.noise_scheduler.get_velocity(batch['next_state_surface'], surface_noise, timesteps)
                if(self.backbone.plev):
                    target_level = self.noise_scheduler.get_velocity(batch['next_state_level'], level_noise, timesteps)
    
            elif self.noise_scheduler.config.prediction_type == "sample":
                target_surface = batch['next_state_surface']
                target_level = batch['next_state_level']





        
        
        # create uncond
        sel = (torch.rand((bs,), device=device) > self.p_uncond)
        pred = self.forward(batch, timesteps,sel[:,None,None,None])
        #print(pred['next_state_surface'].shape)
        #print('prediction',pred['next_state_surface'][0,:,0,0])
        # compute loss
        batch['next_state_surface'] = target_surface
        if(self.backbone.plev):
            batch['next_state_level'] = target_level

        loss = self.loss(pred, batch)
        loss = loss.mean()
        self.mylog(loss=loss)

        return loss


    
    def validation_step(self, batch, batch_nb):        
        # for the validation, we make some generations and log them 
        samples = [self.sample(batch, num_inference_steps=self.num_inference_steps, 
                                denormalize=False, 
                                scheduler=self.scheduler, 
                                seed = j)
                            for j in range(self.num_members)]
        denorm_samples = [self.dataset.denormalize(x) for x in samples]
        denorm_batch = self.dataset.denormalize(batch)
        #tas_image = wandb.Image(samples[0]['state_surface'][0,0], caption="tas")
        images = [bw_to_bwr(x['state_surface'][0,0]) for x in samples]

        #need to fix this
        #self.logger.log_image('tas_image',images=images)
        
        self.metrics.update(denorm_batch,denorm_samples)
        return None


    
    def on_validation_epoch_end(self):
        for metric in [self.metrics]:
            out = metric.compute()
            self.log_dict(out)# dont put on_epoch = True here
        #    print(out)
            metric.reset()


    def test_step(self, batch, batch_nb):
        self.validation_step(batch,batch_nb)

    def on_test_epoch_end(self):
        for metric in [self.metrics]:
            out = metric.compute()
            self.log_dict(out)# dont put on_epoch = True here
        #    print(out)
            metric.reset()



    def predict_step(self,batch,batch_nb):
        #only do rollout for every beginning of year
        device = batch['state_surface'].device
        # print(self.dataset.files)
        # print(self.dataset.id2pt)
        # print(len(self.dataset.id2pt))
        # print(self.dataset.timestamps)
        # print(batch['time'])
        if(int(batch['time'][0].split('-')[-1]) != 2):
            return
        out_dir = f'./plots'

        rollout_length = self.num_rollout_steps    #no greater than 118 please
        rollout_ensemble = []

        for i in range(self.num_members):
            
            rollout = self.sample_rollout(batch,rollout_length=rollout_length,seed = i)
            
            rollout['state_surface'] = torch.stack(rollout['state_surface'])
            rollout['state_surface'] = torch.where(rollout['state_surface']==100,0,rollout['state_surface'])
            rollout_ensemble.append(rollout)
        ipsl_ensemble = []
        batch_timeseries = {'state_surface':[]}


        for i in range(0,self.num_members):
            for j in range(0,rollout_length):
                batch = self.dataset.__getitem__(i*rollout_length + j)
                batch_timeseries['state_surface'].append(batch['state_surface'])
            batch_timeseries['state_surface'] = torch.stack(batch_timeseries['state_surface'])
            ipsl_ensemble.append(batch_timeseries)
            batch_timeseries = {'state_surface':[]}  
        ipsl_ensemble = np.stack(ipsl_ensemble) 


        minimum = 1000
        maximum = -1000
        for var_num in range(0,34):
            fig, axes = plt.subplots(2, figsize=(16, 6))
            axes = axes.flatten()
            for i in rollout_ensemble:
                axes[0].plot(torch.mean(i['state_surface'][:rollout_length,0,var_num],axis=(-1,-2)).cpu())
                local_min = torch.mean(i['state_surface'][:rollout_length,0,var_num],axis=(-1,-2)).min().cpu()
                local_max = torch.mean(i['state_surface'][:rollout_length,0,var_num],axis=(-1,-2)).max().cpu()
                minimum = minimum if minimum < local_min else local_min
                maximum = maximum if maximum > local_max else local_max
            for i in ipsl_ensemble:
                axes[1].plot(torch.nanmean(i['state_surface'][:rollout_length,var_num],axis=(-1,-2)).cpu())
            local_min = torch.mean(ipsl_ensemble[0]['state_surface'][:rollout_length,var_num],axis=(-1,-2)).min().cpu() 
            local_max = torch.mean(ipsl_ensemble[0]['state_surface'][:rollout_length,var_num],axis=(-1,-2)).max().cpu()
            minimum = minimum if minimum < local_min else local_min
            maximum = maximum if maximum > local_max else local_max
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
            if(var_num < 10):
                plt.title(self.dataset.surface_variables[var_num])
            fig.savefig(file_name)
            print(file_name)
        
        
            ds = xr.open_dataset(self.dataset.files[0])
            shell = ds.isel(time=0)
            fig, axes = plt.subplots(1,2, figsize=(16, 6))
            axes = axes.flatten()
            container = []
            for time_step in range(rollout_length):
                # print(np.stack(ensembles[0]['state_surface']).shape)
                # print(np.stack(ipsl_ensemble[0]['state_surface']).shape)
                shell['tas'].data = rollout_ensemble[0]['state_surface'][time_step][0][var_num].cpu()
                   # line = ax1.pcolormesh(steps[time_step][0,0,0])
                line = shell['tas'].plot.pcolormesh(ax=axes[0],add_colorbar=False)
                shell['tas'].data = ipsl_ensemble[0]['state_surface'][time_step][var_num].cpu()
                line1 = shell['tas'].plot.pcolormesh(ax=axes[1],add_colorbar=False)
                title = axes[0].text(0.5,1.05,"Diffusion Step {}".format(time_step), 
                                size=plt.rcParams["axes.titlesize"],
                                ha="center", transform=axes[0].transAxes,)
                axes[0].set_title('Predicted')
                axes[1].set_title('IPSL')
            
                container.append([line,line1,title])
            if(var_num < 10):
                plt.title(self.dataset.surface_variables[var_num])
            
            ani = animation.ArtistAnimation(fig, container, interval=100, blit=True)
            ani.save(f'{out_dir}/diffusion_comparison_{var_num}.gif')
            
    
            #calculate and plot sss and crps for timeseries
            #denormalize
            batch_denormed = []
            denormed_surface_ensembles = []
            denormed_batch_surface_ensembles = []
            denormed_plev_ensembles = []
            denormed_batch_plev_ensembles = []
            for i in range(self.num_members):
                month_index = 0
                denormalized_surface = []
                denormalized_plev = []
                batch_denormed_surface = []
                batch_denormed_plev = []
            
                for index in range(rollout_length):
               # for index in range(len(ensembles[i]['state_surface'])-2):
                    denorm_surface = lambda x,month_index: x[:10].to(device)*torch.tensor(self.dataset.surface_stds[month_index]).to(device) + torch.tensor(self.dataset.surface_means[month_index]).to(device)
                    denorm_plev = lambda x,month_index: x[10:].reshape(-1,8,3,143,144).to(device)*torch.tensor(self.dataset.plev_stds[month_index]).to(device) + torch.tensor(self.dataset.plev_means[month_index]).to(device)
                   # if(cfg.dataloader.flattened_plev):
            #        print(ensembles[i]['state_surface'][index].shape)
             #       print(ipsl_ensemble[i]['state_surface'][index].shape)
                    if(self.dataset.flattened_plev):
                        denormalized_plev.append(denorm_plev(rollout_ensemble[i]['state_surface'][index][0],month_index))
                        batch_denormed_plev.append(denorm_plev(torch.Tensor(ipsl_ensemble[i]['state_surface'][index]),month_index))
            
                    denormalized_surface.append(denorm_surface(rollout_ensemble[i]['state_surface'][index][0],month_index))
                    batch_denormed_surface.append(denorm_surface(torch.Tensor(ipsl_ensemble[i]['state_surface'][index]),month_index))
                    if(month_index == 11):
                        month_index = 0
                    else:
                        month_index += 1
                denormed_surface_ensembles.append(denormalized_surface)
                denormed_batch_surface_ensembles.append(batch_denormed_surface)
            # denormed_surface_variables = torch.stack(denormed_surface_ensembles)
            # denormed_batch_surface_ensembles = torch.stack(denormed_batch_surface_ensembles)
        # print(denormed_surface_ensembles[0].shape)
        # print(denormed_batch_surface_ensembles[0].shape)
        self.metrics.update(denormed_surface_ensembles,denormed_batch_surface_ensembles)
        for metric in [self.metrics]:
            out = metric.compute()
           # self.log_dict(out)# dont put on_epoch = True here
            print(out)
            metric.reset()






    
        
    def loss(self, pred, batch, lat_coeffs=None):
        if lat_coeffs is None:
            lat_coeffs = self.dataset.lat_coeffs_equi
        device = batch['next_state_surface'].device
        
        mask = batch['next_state_surface'] == 20


        
        mse_surface = (pred['next_state_surface'][~mask].squeeze() - batch['next_state_surface'][~mask].squeeze()).pow(2)
        mse_surface = mse_surface.mul(lat_coeffs.to(device)) # latitude coeffs



        if(self.backbone.plev):    
            mse_level = (pred['next_state_level'].squeeze() - batch['next_state_level'].squeeze()).pow(2)
            if mse_level.shape[-2] == 128:
                mse_level = mse_level[..., 4:-4, 8:-8]
            mse_level = mse_level.mul(lat_coeffs.to(device))
            mse_level_w = mse_level.mul(level_coeffs.to(device))
    
        if(self.backbone.plev):
            loss = (mse_surface.sum(1).mean((-3, -2, -1)) + 
                mse_level_w.sum(1).mean((-3, -2, -1)))
        else:
            loss = mse_surface.sum(1).mean((-3, -2, -1))
        return loss


    
    def sample_rollout(self, batch, rollout_length, seed,**kwargs):
        device = batch['state_surface'].device
        history = dict(state_surface=[],next_state_surface=[])
        next_time = batch['next_time']    
        inc_time_vec = np.vectorize(inc_time)
        nulls = torch.where(batch['state_surface']==100,1.0,0.0)        
        print(nulls.shape)
        print(nulls.mean(axis=(0,2,3)))
        for i in range(rollout_length):
            print(batch['time'])
            print(batch['next_time'])
            start = time.time()
           # print(batch['state_surface'].shape)
            print(i,'lead_time')
            sample = self.sample(batch, denormalize=False,num_inference_steps=self.num_inference_steps,scheduler=self.scheduler,seed=100000*seed + i)
            next_time = batch['next_time']    
            cur_year_index = int(next_time[0].split('-')[0]) - 1960
            cur_month_index = int(next_time[0].split('-')[-1]) - 1
            new_state_surface=sample['next_state_surface'][:,:10].to(device)*self.dataset.surface_delta_stds.to(device).unsqueeze(0) + batch['state_surface'][:,:10].to(device)
            #sample['next_state_surface'][:,9,:,:] = torch.where(~self.dataset.ocean_mask.to(device),sample['next_state_surface'][:,9,:,:],torch.nan)
            assert len(sample['next_state_surface'][:,:10].shape) == len(self.dataset.surface_delta_stds.to(device).unsqueeze(0).shape)
            if(self.dataset.flattened_plev):
                new_state_plev=sample['next_state_surface'][:,10:].to(device)*self.dataset.plev_delta_stds.to(device).unsqueeze(0).reshape(1,8*3,1,1) + batch['state_surface'][:,10:].to(device)
                new_state_surface = torch.concatenate([new_state_surface,new_state_plev],axis=1)
            # print(self.dataset.ocean_mask.shape)
            # print(new_state_surface.shape)
            
            # sample['next_state_surface'][:,1,:,:] = torch.where(self.dataset.land_mask.to(device),sample['next_state_surface'][:,1,:,:],0)
            # sample['next_state_surface'][:,2,:,:] = torch.where(self.dataset.land_mask.to(device),sample['next_state_surface'][:,2,:,:],0)
            # sample['next_state_surface'][:,3,:,:] = torch.where(self.dataset.land_mask.to(device),sample['next_state_surface'][:,3,:,:],0)
            # sample['next_state_surface'][:,4,:,:] = torch.where(self.dataset.land_mask.to(device),sample['next_state_surface'][:,4,:,:],0)
            # sample['next_state_surface'][:,6,:,:] = torch.where(self.dataset.land_mask.to(device),sample['next_state_surface'][:,6,:,:],0)
            # sample['next_state_surface'][:,7,:,:] = torch.where(self.dataset.land_mask.to(device),sample['next_state_surface'][:,7,:,:],0)
            # sample['next_state_surface'][:,9,:,:] = torch.where(~self.dataset.ocean_mask.to(device),sample['next_state_surface'][:,9,:,:],0)
            # sample['next_state_surface'][:,10:,:,:] = torch.where(self.dataset.plev_mask.to(device),sample['next_state_surface'][:,10:,:,:],0)
            
            # batch['state_surface'][:,1,:,:] = torch.where(self.dataset.land_mask.to(device),batch['state_surface'][:,1,:,:],0)
            # batch['state_surface'][:,2,:,:] = torch.where(self.dataset.land_mask.to(device),batch['state_surface'][:,2,:,:],0)
            # batch['state_surface'][:,3,:,:] = torch.where(self.dataset.land_mask.to(device),batch['state_surface'][:,3,:,:],0)
            # batch['state_surface'][:,4,:,:] = torch.where(self.dataset.land_mask.to(device),batch['state_surface'][:,4,:,:],0)
            # batch['state_surface'][:,6,:,:] = torch.where(self.dataset.land_mask.to(device),batch['state_surface'][:,6,:,:],0)
            # batch['state_surface'][:,7,:,:] = torch.where(self.dataset.land_mask.to(device),batch['state_surface'][:,7,:,:],0)
            # batch['state_surface'][:,9,:,:] = torch.where(~self.dataset.ocean_mask.to(device),batch['state_surface'][:,9,:,:],0)
            # batch['state_surface'][:,10:,:,:] = torch.where(self.dataset.plev_mask.to(device),batch['state_surface'][:,10:,:,:],0)
            # batch['state_surface'] = torch.where(nulls==1.0,100,batch['state_surface'])
            # new_state_surface = torch.where(nulls==1.0,100,new_state_surface)
            print(batch['state_surface'][0,0].mean())
            print(batch['state_surface'][0,11].mean())
            # print(new_state_surface[:,9,:,:].shape)
            history['next_state_surface'].append(new_state_surface)
            history['state_surface'].append(batch['state_surface'])
            batch = dict(state_surface=new_state_surface,
                         prev_state_surface=batch['state_surface'],
                               #state_level=denorm_sample['next_state_level'],
                             #  next_state_surface=batch['next_state_surface'],
                               time=next_time,
                               state_constant=batch['state_constant'],
                                 forcings=torch.Tensor(self.dataset.atmos_forcings[:,cur_year_index]).unsqueeze(0).to(device), #I have no idea why i need to unsqueeze here does lightning add a dimenions with the batch? who knows
                                 state_depth=torch.empty(0),
                                 solar_forcings=torch.Tensor(self.dataset.solar_forcings[cur_year_index,cur_month_index]).unsqueeze(0).to(device),
                               next_time=inc_time_vec(next_time))
            end = time.time()
            print(f'sample took {end - start} to run')
        return history

    
    
    def sample(self, batch, 
                scheduler,
                num_inference_steps,
                denormalize,
                seed=0,
                cf_guidance=None,
                ):
        if cf_guidance is None:
            cf_guidance = 1
        if scheduler == 'ddpm':
            scheduler = deepcopy(self.noise_scheduler)
        elif scheduler == 'ddim':
            from diffusers import DDIMScheduler
            sched_cfg = self.noise_scheduler.config
            scheduler = DDIMScheduler(num_train_timesteps=sched_cfg.num_train_timesteps,
                                                       beta_schedule=sched_cfg.beta_schedule,
                                                       prediction_type=sched_cfg.prediction_type,
                                                       beta_start=sched_cfg.beta_start,
                                                       beta_end=sched_cfg.beta_end,
                                                       clip_sample=False,
                                                       clip_sample_range=3.5,
                                                       rescale_betas_zero_snr=True,

                                               #      rescale_betas_zero_snr=True,
                                                     #  timestep_spacing='trailing',
                                                      # thresholding=True,
                                    #  dynamic_thresholding_ratio=0.70
                                                     )
        elif scheduler == 'flow':
            scheduler = deepcopy(self.noise_scheduler)


        scheduler.set_timesteps(num_inference_steps)
        #mask = torch.where(batch['state_surface']==20,1.0,0.0)        

        local_batch = {k:v for k, v in batch.items() if not k.startswith('next')} # ensure no data leakage
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)
        surface_noise = torch.randn(local_batch['state_surface'].size(), generator=generator)
        local_batch['surface_noisy'] = surface_noise.to(self.device)
        if(self.backbone.plev):
            level_noise = torch.randn_like(batch['next_state_level'])
            local_batch['level_noisy'] = surface_noise.to(self.device)
        steps = []
        with torch.no_grad():
            for t in scheduler.timesteps:
                pred = self.forward(local_batch, torch.tensor([t]).to(self.device))
                #scheduler step finds the actual output (in this case, the delta) where next_state_surface is the velocity predicted
                #uses https://github.com/huggingface/diffusers/blob/668e34c6e0019b29c887bcc401c143a7d088cb25/src/diffusers/schedulers/scheduling_ddpm.py#L525


                # def remove_outliers(t:torch.Tensor):
                    
                #     #REMOVE OUTLIERS
                #     maximum = torch.quantile(t.reshape(34,-1),0.999,dim=1)
                #     minimum = torch.quantile(t.reshape(34,-1),0.001,dim=1)
                #     maximum = maximum.unsqueeze(1).unsqueeze(2).expand(-1,143,144)
                #     minimum = minimum.unsqueeze(1).unsqueeze(2).expand(-1,143,144)
                #     t = torch.clamp(t,min=minimum,max=maximum)
                #     return t
                
                # pred['next_state_surface'] = remove_outliers(pred['next_state_surface'])

                
                local_batch['surface_noisy'] = scheduler.step(pred['next_state_surface'], t, 
                                                            local_batch['surface_noisy'], 
                                                            generator=generator).prev_sample
                # original_sample = scheduler.step(pred['next_state_surface'], t, 
                #                                             local_batch['surface_noisy'], 
                #                                             generator=generator).pred_original_sample
                # if(self.backbone.plev):
                #     local_batch['level_noisy'] = scheduler.step(pred['next_state_level'], t, 
                #                                                 local_batch['level_noisy'], 
                #                                                 generator=generator).prev_sample

               # local_batch['surface_noisy'] = torch.where(mask==1,100,local_batch['surface_noisy'])
               #  local_batch['surface_noisy'] = remove_outliers(local_batch['surface_noisy'])
               # steps.append(local_batch['surface_noisy'])
                steps.append(local_batch['surface_noisy'])
              #  print(local_batch['surface_noisy'].shape)
              #  print(pred['next_state_surface'].shape)
                #(alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
               # pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output

               # local_batch['surface_noisy'] = (((alpha_prod_t**0.5) * local_batch['surface_noisy']) - vel) /(beta_prod_t**0.5)
                #due to the way the model is set up, the state_surface is modified in the forward loop
                # print('local_batch',local_batch['state_surface'].shape,local_batch['surface_noisy'].shape)
                if(self.dataset.flattened_plev):
                    local_batch['state_surface'] = local_batch['state_surface'][:,:34]
                else:
                    local_batch['state_surface'] = local_batch['state_surface'][:,:10]
                # if(self.backbone.plev):
                #     local_batch['state_level'] = local_batch['state_level'][:,:8]
                #     sample = dict(next_state_surface=local_batch['surface_noisy'],state_surface=local_batch['state_surface'],next_state_level=local_batch['level_noisy'])
                #else:
            sample = dict(next_state_surface=local_batch['surface_noisy'],state_surface=local_batch['state_surface'],time=batch['time'],next_time=batch['next_time']) 
        #visualize_denoise(steps,self.dataset)
        if denormalize:
            #sample,batch = self.dataset.denormalize(sample, batch)
            pass
        return sample


    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), 
                                lr=self.lr, betas=self.betas, 
                                weight_decay=self.weight_decay)
        sched = diffusers.optimization.get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
            num_cycles=self.num_cycles)
        sched = { 'scheduler': sched,
                        'interval': 'step', # or 'epoch'
                        'frequency': 1}
        return [opt], [sched]