import lightning.pytorch as pl
import diffusers
import torch
from ipsl_dcpp.model.pangu import TimestepEmbedder
from ipsl_dcpp.evaluation.metrics import EnsembleMetrics
import datetime
import numpy as np
import time
import os 
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib import animation
from copy import deepcopy
from ipsl_dcpp.utils.visualization_utils import make_gif,el_nino_34_index,plot_with_fill
from pathlib import Path

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
        num_batch_examples,
        scheduler,
        elevation,
        month_embed,
        lat_weight
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
        self.num_batch_examples = num_batch_examples
        self.elevation = elevation
        self.month_embed = month_embed
        self.lat_weight = lat_weight
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
        device = batch['state_surface'].device
        if(self.elevation):
            batch['state_surface'] = torch.cat([batch['state_surface']*sel,batch['prev_state_surface']*sel, 
                                   batch['surface_noisy'],batch['state_constant']], dim=1)
        else:
            batch['state_surface'] = torch.cat([batch['state_surface']*sel,batch['prev_state_surface']*sel, 
                                   batch['surface_noisy']], dim=1)
        if(self.backbone.plev):
            batch['state_level'] = torch.cat([batch['state_level']*sel,batch['prev_state_level']*sel, 
                                   batch['level_noisy']], dim=1)
        # print('batch_time',batch['time'])


        year = torch.tensor([int(x[0:4]) for x in batch['time']]).to(device)
        year_emb = self.month_embedder(year)
        timestep_emb = self.timestep_embedder(timesteps)
        ch4_emb = self.timestep_embedder(batch['forcings'][:,0])
        cfc11_emb = self.timestep_embedder(batch['forcings'][:,1])
        cfc12_emb = self.timestep_embedder(batch['forcings'][:,2])
        c02_emb = self.timestep_embedder(batch['forcings'][:,3])
        if(self.month_embed):
            month = torch.tensor([int(x[5:7]) for x in batch['time']]).to(device)
            month_emb = self.month_embedder(month)
            cond_emb = (month_emb + year_emb + timestep_emb + ch4_emb + cfc11_emb + cfc12_emb + c02_emb)
        else:
            cond_emb = (year_emb + timestep_emb + ch4_emb + cfc11_emb + cfc12_emb + c02_emb)
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
            
            batch['surface_noisy'] = sigmas * surface_noise + (1.0 - sigmas) * batch['next_state_surface']
          #  batch['level_noisy'] = noisy_level_input = sigmas * level_noise + (1.0 - sigmas) * batch['next_state_level']

            target_surface = surface_noise - batch['next_state_surface']
         #   target_level = level_noise - batch['next_state_level']
        else:             
            batch['surface_noisy'] = self.noise_scheduler.add_noise(batch['next_state_surface'], surface_noise, timesteps)
            batch['surface_noisy'] = self.noise_scheduler.scale_model_input(batch['surface_noisy'])
            if(self.backbone.plev):
                level_noise = torch.randn_like(batch['next_state_level'])
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
        #in validation, we take ONE batch and put it with X num_members
        #this is different from predict, where we take x num_members and put it with 1 ensemble each - need to change this
        
        samples = [self.sample(batch, num_inference_steps=self.num_inference_steps, 
                                denormalize=False, 
                                scheduler=self.scheduler, 
                                seed = j)
                            for j in range(self.num_members)]
        denorm_samples = [self.dataset.denormalize(x)['next_state_surface'] for x in samples]
        denorm_batch = self.dataset.denormalize(batch)


       # denorm_samples [y for y in denorm_samples]
        denorm_samples = torch.stack(denorm_samples)
        denorm_samples = denorm_samples.swapdims(0,1)
 


        #take first batch and add num_members dim, num_ensembles already populated
        denorm_samples = denorm_samples[:1]
        #take first instance of batch only, and unsqueeze for num_ensembles_dim
        denorm_batch = denorm_batch['next_state_surface'][:1].unsqueeze(0)
        self.metrics.update(denorm_batch,denorm_samples)

        #include image?
        #tas_image = wandb.Image(samples[0]['state_surface'][0,0], caption="tas")
       # images = [bw_to_bwr(x['state_surface'][0,0]) for x in samples]
        #self.logger.log_image('tas_image',images=images)
        return None


    
    def on_validation_epoch_end(self):
        for metric in [self.metrics]:
            out = metric.compute()
            self.log_dict(out)# dont put on_epoch = True here
            print(out)
            metric.reset()


    # def test_step(self, batch, batch_nb):
    #     self.validation_step(batch,batch_nb)

    # def on_test_epoch_end(self):
    #     for metric in [self.metrics]:
    #         out = metric.compute()
    #         self.log_dict(out)# dont put on_epoch = True here
    #     #    print(out)
    #         metric.reset()

    def test_step(self,batch,batch_nb):
        out_dir = f'./plots/{self.dataset.plot_output_path}_long_rollout/'
        os.makedirs(out_dir,exist_ok=True)
        for ensemble_member in range(0,7):
            rollout = self.sample_rollout(batch,rollout_length=118*5,seed = ensemble_member)
            torch.save(torch.stack(rollout['state_surface']),f'{out_dir}/long_rollout_{ensemble_member}.pt')


    def unnormalize_data(self,data):
        #denormalize, this is different from the denorm function in the IPSL_DATASET class as it does not consider dictionary of 'state_surface', should this be a function as well? 
        device = 'cuda:0'
        denormed_surface_ensembles = []
        denormed_batch_surface_ensembles = []
        for ic_index in range(self.num_batch_examples):
            month_index = 1
            batch_denormalized = []
            denormed_surface_members = []
            for rollout_index in range(self.num_rollout_steps):
                #denormalize batch
                batch_denormed_surface = self.dataset.denorm_surface_variables(
                    data[None,0,ic_index,0,rollout_index],month_index,device)
                batch_denormed_plev = self.dataset.denorm_plev_variables(
                    data[None,0,ic_index,0,rollout_index],month_index,device)
                batch_denormalized.append(torch.concatenate([batch_denormed_surface,batch_denormed_plev],axis=1))
                denormalized_surface = []
                for member_index in range(self.num_members):
                    
                    denormed_plev = self.dataset.denorm_plev_variables(
                        data[None,1,ic_index,member_index,rollout_index],month_index,device
                    ) 
                    denormed_surface = self.dataset.denorm_surface_variables(
                        data[None,1,ic_index,member_index,rollout_index],month_index,device
                    )
                    denormalized_surface.append(torch.concatenate([denormed_surface,denormed_plev],axis=1))
                if(month_index == 11):
                    month_index = 0
                else:
                    month_index += 1
                denormed_surface_members.append(torch.stack(denormalized_surface))
            denormed_surface_ensembles.append(torch.stack(denormed_surface_members))
            denormed_batch_surface_ensembles.append(torch.stack(batch_denormalized))
        # print('pred_denorm',torch.stack(denormed_surface_ensembles).shape)
        # print(torch.stack(denormed_batch_surface_ensembles).shape)
        
        denormed_surface_ensembles = torch.stack(denormed_surface_ensembles).squeeze(3)
        denormed_surface_ensembles = torch.swapdims(denormed_surface_ensembles,1,2)
        denormed_batch_surface_ensembles = torch.stack(denormed_batch_surface_ensembles)
        denormed_batch_surface_ensembles = torch.swapdims(denormed_batch_surface_ensembles,1,2)
        denormed_batch_surface_ensembles = denormed_batch_surface_ensembles.expand(-1,self.num_ensemble_members,-1,-1,-1,-1)
        
        #should be same shape as normed data: [[batch,pred],num_batch_examples (diff IC), num_members, rollout_length,var,lat,lon]
        denormed_data = torch.stack([denormed_batch_surface_ensembles,denormed_surface_ensembles])
        return denormed_data 

    def make_rollout_and_batch_data(self):
        #returns tensor of shape [[batch,pred],num_batch_examples (diff IC), num_members, rollout_length,var,lat,lon]
        ipsl_ensemble = []
        rollout_ensemble = []
        device = 'cuda:0'
        seed_id = 0 
        for k in range(0,self.num_batch_examples):
            batch_timeseries = []
            for j in range(0,self.num_rollout_steps):
                batch = self.dataset.__getitem__((k*118) + j)
                batch_timeseries.append(batch['state_surface'])
                print(batch['time'])
                if(j == 0):
                    batch = {k:[batch[k]] if k == 'time' or k == 'next_time' else batch[k].unsqueeze(0) for k in batch.keys()}  #simulate lightnings batching dimension
                    batch['state_surface'] = batch['state_surface'].to(device)
                    batch['prev_state_surface'] = batch['prev_state_surface'].to(device)
                    batch['forcings'] = batch['forcings'].to(device)
                    batch['solar_forcings'] = batch['solar_forcings'].to(device)#make n for each batch, could do experiments for different num per batch
                    rollout_members = []
                    for i in range(0,self.num_members):
                        batch['state_constant'] = batch['state_constant'].to(device)
                     #   rollout = self.sample_rollout(batch,rollout_length=rollout_length,seed = i)
                        rollout = self.sample_rollout(batch,rollout_length=self.num_rollout_steps,seed = seed_id)
                        seed_id = seed_id + 1
                        rollout_members.append(torch.stack(rollout['state_surface']))
            rollout_ensemble.append(torch.stack(rollout_members))
            ipsl_ensemble.append(torch.stack(batch_timeseries))
        ipsl_ensemble = torch.stack(ipsl_ensemble)
        rollout_ensemble = torch.stack(rollout_ensemble)
        #some rejigging of the shapes 
        rollout_ensemble = rollout_ensemble.squeeze(3)
        ipsl_ensemble = ipsl_ensemble.unsqueeze(1).to(device)
        ipsl_ensemble = ipsl_ensemble.expand(-1,self.num_ensemble_members,-1,-1,-1,-1) # this doubles the "member" dimension to match rollout and make it stackable, is this too big of a waste of space ? 
        data = torch.stack([ipsl_ensemble,rollout_ensemble])    
        return data


    def sharpness_test(self):
        #returns sharpness for rollout_length
        data = self.make_rollout_and_batch_data()
        denormed_data = self.unnormalize_data(data)
        device = 'cuda:0'
        sharpness = []
        for ic_index in range(self.num_batch_examples):
            self.metrics = EnsembleMetrics(dataset=self.dataset).to(device)
            output_metrics = []
            for rollout_index in range(self.num_rollout_steps):
                #needs to be [1, num_pred_members_per_sample (should be 1 for batch), variables, lat, lon]
                self.metrics.update(
                    denormed_data[None,0,ic_index,:1,rollout_index],
                    denormed_data[None,1,ic_index,:,rollout_index]
                )
                for metric in [self.metrics]:
                    out = metric.compute()
                    metric.reset()
                    output_metrics.append(torch.tensor(list(out.values())))
            sharpness.append(output_metrics)
        return sharpness

    
    def predict_step(self,batch,batch_nb):        
        #make variable names for output
        var_names = self.dataset.surface_variables.copy()
        plev_var_names = [[x+'_850',x+'_750',x+'_500'] for x in self.dataset.plev_variables]
        for x in plev_var_names:
            for name in x:
                var_names.append(name)
        var_names = [(var_names[index], 0, index) for index in range(len(var_names))]
        device = batch['state_surface'].device
        out_dir = f'./plots/{self.dataset.plot_output_path}/'
        os.makedirs(out_dir,exist_ok=True)
        batch_colors = ['blue','green','black']
        data = self.make_rollout_and_batch_data()
        #shape is now [[batch,pred],num_batch_examples (diff IC), num_members, rollout_length,var,lat,lon]
        # print(data.shape)

        #make plots for each variable in the set
        for var_num in range(0,34):
            fig, axes = plt.subplots(7, figsize=(16, 16))
            axes = axes.flatten()


            
            #make plot of actual output and compare
            minimum = torch.min(torch.mean(data[:,:,:,:,var_num],axis=(-1,-2))).cpu()
            maximum = torch.max(torch.mean(data[:,:,:,:,var_num],axis=(-1,-2))).cpu()
            for ic_index in range(self.num_batch_examples):
                for member_index in range(self.num_members): 
                    means = torch.mean(data[1,ic_index,member_index,:,var_num],axis=(-1,-2)) 
                    axes[0].plot(means.cpu(),color=batch_colors[ic_index])
                axes[1].plot(torch.mean(data[0,ic_index,0,:,var_num],axis=(-1,-2)).cpu(),color=batch_colors[ic_index])
            axes[0].set_ylim(minimum,maximum)
            axes[1].set_ylim(minimum,maximum)
            axes[0].set_title('Predicted')
            axes[1].set_title('IPSL')
            axes[0].set_ylabel(f'Normalized Value')
            #axes[1].set_ylabel(f'Normalized {var_names[var_num]} Value')

            
            #make gif of only one IC
            make_gif(
                    data=data[:,0,0,:,var_num].cpu(),
                    rollout_length=self.num_rollout_steps,
                    var_name=var_names[var_num],
                    file_name=f'{out_dir}/normalized_diffusion_comparison_{var_names[var_num][0]}_ffmpeg',
                    save=True,
                                ffmpeg=True

            )

            denormed_data = self.unnormalize_data(data)

            #calculate and plot sss and crps for timeseries
            for ic_index in range(self.num_batch_examples):
                self.metrics = EnsembleMetrics(dataset=self.dataset).to(device)
                output_metrics = []
                for rollout_index in range(self.num_rollout_steps):
                    #needs to be [1, num_pred_members_per_sample (should be 1 for batch), variables, lat, lon]
                    self.metrics.update(
                        denormed_data[None,0,ic_index,:1,rollout_index],
                        denormed_data[None,1,ic_index,:,rollout_index]
                    )
                    for metric in [self.metrics]:
                        out = metric.compute()
                        metric.reset()
                        output_metrics.append(torch.tensor(list(out.values())))
                        
                output_metrics_tensor = torch.stack(output_metrics)
                axes[2].plot(output_metrics_tensor[:,5*(var_num)].cpu(),color=batch_colors[ic_index])
                axes[2].set_title(list(out.keys())[5*(var_num)])
                axes[3].plot(output_metrics_tensor[:,2+ (5*var_num)].cpu(),color=batch_colors[ic_index])
                axes[3].set_title(list(out.keys())[2+(5*var_num)])
                axes[4].plot(output_metrics_tensor[:,3+(5*var_num)].cpu(),color=batch_colors[ic_index])
                axes[4].set_title(list(out.keys())[3+(5*var_num)])
                axes[5].plot(output_metrics_tensor[:,4+(5*var_num)].cpu(),color=batch_colors[ic_index])
                axes[5].set_title(list(out.keys())[4+(5*var_num)])
                fig.tight_layout()


#[[batch,pred],num_batch_examples (diff IC), num_members, rollout_length,var,lat,lon]
            #plot bias 
            for ic_index in range(self.num_batch_examples):
                pred_means = denormed_data[1,ic_index,:,:,var_num].mul(self.dataset.lat_coeffs_equi[None:,:,:].to(device)).mean(axis=(-4,-2,-1)).squeeze()
                batch_means = denormed_data[0,ic_index,0,:,var_num].mul(self.dataset.lat_coeffs_equi[:,:,:].to(device)).mean(axis=(-1,-2)).squeeze()
                pred_stds = pred_means.std(axis=(0))
                batch_stds = batch_means.std(axis=(0))
                plot_with_fill(pred_means, pred_stds, batch_means,batch_stds,axes=axes[6])
                axes[6].set_title('unnormalized comp')

                
            
            #plot elnino34 indices 

            if(var_num == 9): #aka its top of surface sea temp
                tos_fig,tos_axes = plt.subplots(2,figsize=(16,16))
                tos_axes = tos_axes.flatten()
                batch_el_nino_34 = el_nino_34_index(denormed_data[0].cpu())
    
                tos_axes[0].plot(batch_el_nino_34,color='black')
                tos_axes[0].axhline(0, color='black', lw=0.5)
                tos_axes[0].axhline(0.4, color='black', linewidth=0.5, linestyle='dotted')
                tos_axes[0].axhline(-0.4, color='black', linewidth=0.5, linestyle='dotted')
                tos_axes[0].set_title('IPSL Niño 3.4 Index');
    
                predicted_el_nino_34 = el_nino_34_index(denormed_data[1].cpu())       
    
                tos_axes[1].plot(predicted_el_nino_34,color='black')
                tos_axes[1].axhline(0, color='black', lw=0.5)
                tos_axes[1].axhline(0.4, color='black', linewidth=0.5, linestyle='dotted')
                tos_axes[1].axhline(-0.4, color='black', linewidth=0.5, linestyle='dotted')
                tos_axes[1].set_title('Rollout Niño 3.4 Index');
                tos_fig.savefig(f'{out_dir}/el_nino_index.png')
            
            file_name = f'{out_dir}/statistics_for_{var_names[var_num][0]}.png'
            output_file = Path(file_name)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            fig.savefig(file_name)
        
            make_gif(
                    data=denormed_data[:,0,0,:,var_num].cpu(),
                    rollout_length=self.num_rollout_steps,
                    var_name=var_names[var_num],
                    file_name=f'{out_dir}/denormalized_diffusion_comparison_{var_names[var_num][0]}_ffmpeg',
                    save=True,
                denormalized=True,
                ffmpeg=True
            )
            return 
    
        
    def loss(self, pred, batch, lat_coeffs=None):
        if lat_coeffs is None:
            lat_coeffs = self.dataset.lat_coeffs_equi
        device = batch['next_state_surface'].device
        mask = batch['next_state_surface'] != self.dataset.mask_value

        
        print(mask.shape,'mask')
        # print(pred['next_state_surface'].shape)
        # print(pred['next_state_surface'][~mask].shape)
        print((pred['next_state_surface'] * mask).shape)
        mse_surface = ((pred['next_state_surface'] * mask) - (batch['next_state_surface'] * mask)).pow(2)
        if(self.lat_weight):
            mse_surface = mse_surface.mul(lat_coeffs.to(device)) # latitude coeffs
        
       # xx = mse_surface.mul(lat_coeffs.to(device)) # latitude coeffs
        # print(xx.shape)
        print(lat_coeffs.shape)
        print(mse_surface.shape)
        # print(pred['next_state_surface'][~mask].squeeze().shape)
        # print((pred['next_state_surface'].squeeze() - batch['next_state_surface'].squeeze()).pow(2).shape)
        if(self.backbone.plev):    
            mse_level = (pred['next_state_level'].squeeze() - batch['next_state_level'].squeeze()).pow(2)
            if mse_level.shape[-2] == 128:
                mse_level = mse_level[..., 4:-4, 8:-8]
            mse_level = mse_level.mul(lat_coeffs.to(device))
            mse_level_w = mse_level.mul(level_coeffs.to(device))
            loss = (mse_surface.sum(1).mean((-3, -2, -1)) + 
                mse_level_w.sum(1).mean((-3, -2, -1)))
        else:
            loss = mse_surface.sum(0).mean((-3, -2, -1))
            print(loss.shape)
        return loss


    
    def sample_rollout(self, batch, rollout_length, seed,**kwargs):
        device = batch['state_surface'].device
        history = dict(state_surface=[],next_state_surface=[])
        next_time = batch['next_time']    
        inc_time_vec = np.vectorize(inc_time)
        nulls = torch.where(batch['state_surface']==0,1.0,0.0)     
        for i in range(rollout_length):
            start = time.time()
            print('before sampling',batch['time'])
            sample = self.sample(batch, denormalize=False,num_inference_steps=self.num_inference_steps,scheduler=self.scheduler,seed=100000*seed + i)
            next_time = batch['next_time']    
            print('after sampling',batch['time'])
            cur_year_index = int(next_time[0].split('-')[0]) - 1960
            cur_month_index = int(next_time[0].split('-')[-1]) - 1
            
            if(self.dataset.delta):
                new_state_surface=sample['next_state_surface'][:,:10].to(device)*self.dataset.surface_delta_stds.to(device).unsqueeze(0) + batch['state_surface'][:,:10].to(device)
                if(self.dataset.flattened_plev):
                    new_state_plev=sample['next_state_surface'][:,10:].to(device)*self.dataset.plev_delta_stds.to(device).unsqueeze(0).reshape(1,8*3,1,1) + batch['state_surface'][:,10:].to(device)
                    new_state_surface = torch.concatenate([new_state_surface,new_state_plev],axis=1)
            else:
                new_state_surface=sample['next_state_surface']
            batch['state_surface'] = torch.where(nulls==1.0,0,batch['state_surface'])
            new_state_surface = torch.where(nulls==1.0,0,new_state_surface)
            history['next_state_surface'].append(new_state_surface)
            history['state_surface'].append(batch['state_surface'])
            batch = dict(state_surface=new_state_surface,
                         prev_state_surface=batch['state_surface'],
                         time=next_time,
                         state_constant=batch['state_constant'],
                         forcings=torch.Tensor(self.dataset.atmos_forcings[:,cur_year_index]).unsqueeze(0).to(device), #I have no idea why i need to unsqueeze here does lightning add a dimenions with the batch? who knows
                         state_depth=torch.empty(0),
                         solar_forcings=torch.Tensor(self.dataset.solar_forcings[cur_year_index,cur_month_index]).unsqueeze(0).to(device),
                         next_time=inc_time_vec(next_time)
                        )
            end = time.time()
            print(f'sample {i} took {end - start} to run')
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
           # level_noise = torch.randn_like(batch['next_state_level'])
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
                # print(local_batch['surface_noisy'].shape)
                # print(pred['next_state_surface'].shape)

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
