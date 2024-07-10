
import lightning.pytorch as pl
import diffusers
import torch
from ipsl_dcpp.model.pangu import TimestepEmbedder
from ipsl_dcpp.evaluation.metrics import EnsembleMetrics
import datetime
import numpy as np
import time
import pandas as pd
import os 
import pickle
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
        self.scheduler = 'ddpm'
        self.num_members = num_ensemble_members
        self.noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=num_diffusion_timesteps,
                                                       beta_schedule='squaredcos_cap_v2',
                                                       beta_start=0.0001,
                                                       beta_end=0.012,
                                                       prediction_type=prediction_type,
                                                       clip_sample=False,
                                                       clip_sample_range=1e6,
                                                       rescale_betas_zero_snr=True,
                                                       )
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
        #print(sel.shape)
        # batch['state_surface'] = batch['state_surface'].to('cuda')
        # batch['state_constant']= batch['state_constant'].to('cuda')
        # batch['prev_state_surface']=batch['prev_state_surface'].to('cuda')
        device = batch['state_surface'].device

        # print(batch['surface_noisy'].shape)
        # print(batch['state_surface'].shape)
        # print(batch['state_constant'].shape)
        # print(batch['prev_state_surface'].shape)
       # batch['surface_noisy'] = batch['surface_noisy'].squeeze(1)
        # print(batch['state_surface'].device)
        # print(batch['surface_noisy'].device)
        # print(batch['state_constant'].device)
        # print(batch['prev_state_surface'].device)

        batch['state_surface'] = torch.cat([batch['state_surface']*sel,batch['prev_state_surface']*sel, 
                                   batch['surface_noisy'],batch['state_constant']], dim=1)
        if(self.backbone.plev):
            batch['state_level'] = torch.cat([batch['state_level']*sel,batch['prev_state_level']*sel, 
                                   batch['level_noisy']], dim=1)
  #      print(batch['state_surface'].shape, 'concatenated')
        month = torch.tensor([int(x[5:7]) for x in batch['time']]).to(device)
        month_emb = self.month_embedder(month)
        timestep_emb = self.timestep_embedder(timesteps)
     #   print(batch['forcings'].shape,'first')
        ch4_emb = self.timestep_embedder(batch['forcings'][:,0])
        cfc11_emb = self.timestep_embedder(batch['forcings'][:,1])
        cfc12_emb = self.timestep_embedder(batch['forcings'][:,2])
        c02_emb = self.timestep_embedder(batch['forcings'][:,3])

        cond_emb = (month_emb + timestep_emb + ch4_emb + cfc11_emb + cfc12_emb + c02_emb)
        for wavelength_index in range(len(batch['solar_forcings'][0])):
            cond_emb += self.timestep_embedder(batch['solar_forcings'][:,wavelength_index]) 
        out = self.backbone(batch, cond_emb)
    #    print(out['next_state_surface'].shape,'out')
        # if self.noise_scheduler.config.prediction_type == "v_prediction":
        #     # estimate input noise
        #     alpha = self.noise_scheduler.alphas_cumprod.to(self.device)[timesteps][:, None, None, None]

        #     est_surface_noise = (batch['surface_noisy'] - out['next_state_surface']*alpha.sqrt())/(1-alpha).sqrt().add(1e-6)
        #    # est_level_noise = (batch['level_noisy'] - out['next_state_level']*alpha.sqrt())/(1-alpha).sqrt().add(1e-6)

        #     out = dict(next_state_surface=self.noise_scheduler.get_velocity(
        #                      out['next_state_surface'], est_surface_noise, timesteps),
        #                      #next_state_level=self.noise_scheduler.get_velocity(
        #                      #out['next_state_level'], est_level_noise, timesteps)
        #                      )

        return out

    def training_step(self, batch, batch_nb):
        device = batch['state_surface'].device
        bs = batch['state_surface'].shape[0]

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
        surface_noise = torch.randn_like(batch['next_state_surface'])
        batch['surface_noisy'] = self.noise_scheduler.add_noise(batch['next_state_surface'], surface_noise, timesteps)
        batch['surface_noisy'] = self.noise_scheduler.scale_model_input(batch['surface_noisy'])
        if(self.backbone.plev):
            level_noise = troch.randn_like(batch['next_state_level'])
            batch['level_noisy'] = self.noise_scheduler.add_noise(batch['level_noisy'], level_noise,timesteps)
            batch['level_noisy'] = self.noise_scheduler.scale_model_input(batch['level_noisy'])
        
        # Get the target for loss depending on the prediction type
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

    def loss(self, pred, batch, lat_coeffs=None):
        if lat_coeffs is None:
            lat_coeffs = self.dataset.lat_coeffs_equi
        device = batch['next_state_surface'].device
        mse_surface = (pred['next_state_surface'].squeeze() - batch['next_state_surface'].squeeze()).pow(2)
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

    def sample_rollout(self, batch, *args, lead_time_months, seed,**kwargs):
        device = batch['state_surface'].device
        history = dict(state_surface=[],next_state_surface=[])
        next_time = batch['next_time']    
        cur_year_index = int(next_time[0].split('-')[0]) - 1960
        cur_month_index = int(next_time[0].split('-')[-1]) - 1
        inc_time_vec = np.vectorize(inc_time)
        for i in range(lead_time_months):
            start = time.time()
           # print(batch['state_surface'].shape)
            
            print(i,'lead_time')
            
            sample,steps = self.sample(batch, denormalize=False,num_inference_steps=self.num_inference_steps,scheduler='ddim',seed=100000*seed + i)
            next_time = batch['next_time']    
            cur_year_index = int(next_time[0].split('-')[0]) - 1960
            cur_month_index = int(next_time[0].split('-')[-1]) - 1
            new_state_surface=sample['next_state_surface'][:,:10].to(device)*self.dataset.surface_delta_stds.to(device).unsqueeze(0) + batch['state_surface'][:,:10].to(device)
            assert len(sample['next_state_surface'][:,:10].shape) == len(self.dataset.surface_delta_stds.to(device).unsqueeze(0).shape)
            # print(sample['next_state_surface'][:,:10].shape)
            # print(self.dataset.surface_delta_stds.shape)
            # print(new_state_surface.shape)
          #  prev_state_surface=sample['state_surface'].to(device)*torch.unsqueeze(self.dataset.surface_delta_stds,0).to(device) + batch['state_surface'].to(device)
            #need to reshape flattened plev back to level and then un-delta-ize
            if(self.dataset.flattened_plev):
                new_state_plev=sample['next_state_surface'][:,10:].to(device)*self.dataset.plev_delta_stds.to(device).unsqueeze(0).reshape(1,8*3,1,1) + batch['state_surface'][:,10:].to(device)
                # new_state_plev = new_state_plev.reshape(-1,8*3,143,144)
                # print('plev adjust',sample['next_state_surface'][:,10:].to(device).reshape(-1,8,3,143,144).shape)
                # print(new_state_plev.shape)
                # print(new_state_surface.shape)
                # print(batch['state_surface'].shape)
                # print(self.dataset.plev_delta_stds.shape)
                new_state_surface = torch.concatenate([new_state_surface,new_state_plev],axis=1)



            #visualize denoising proccess
            from matplotlib import animation
            import xarray as xr
            import matplotlib.pyplot as plt
            ds = xr.open_dataset(self.dataset.files[0])
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
            ani.save(f'denoise_{var_num}_{i}_epsilon.gif')


            fig, axes = plt.subplots(1,1, figsize=(16, 6))
            var_num = -1
            print(sample['next_state_surface'].shape)
            shell['tas'].data = sample['next_state_surface'][0][var_num].cpu()
           # line = ax1.pcolormesh(steps[time_step][0,0,0])
            line = shell['tas'].plot.pcolormesh(ax=axes,add_colorbar=True,vmax=5,vmin=-5)
            fig.savefig(f'denoised_image_epsilon__{var_num}_{i}.png')



            
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
            scheduler = self.noise_scheduler
        elif scheduler == 'ddim':
            from diffusers import DDIMScheduler
            print('using ddim')
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
            

        scheduler.set_timesteps(num_inference_steps)

        local_batch = {k:v for k, v in batch.items() if not k.startswith('next')} # ensure no data leakage
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)
    
        surface_noise = torch.randn(local_batch['state_surface'].size(), generator=generator)
        local_batch['surface_noisy'] = surface_noise.to(self.device)
        print(surface_noise.shape)
        if(self.backbone.plev):
            level_noise = torch.randn_like(batch['next_state_level'])
            local_batch['level_noisy'] = surface_noise.to(self.device)

        steps = []
        with torch.no_grad():
            for t in scheduler.timesteps:

              #  prev_t = scheduler.previous_timestep(t)

            #    alpha_prod_t = scheduler.alphas_cumprod[t]
              #  alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.one
             #   beta_prod_t = 1 - alpha_prod_t
            #    beta_prod_t_prev = 1 - alpha_prod_t_prev
            #    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
             #   current_beta_t = 1 - current_alpha_t
                # print(t)
              #  print(local_batch['surface_noisy'][0,0,0,:],'next time step')

                #sel = (torch.rand((1,), device='cpu') > self.p_uncond)
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
                original_sample = scheduler.step(pred['next_state_surface'], t, 
                                                            local_batch['surface_noisy'], 
                                                            generator=generator).pred_original_sample
                # if(self.backbone.plev):
                #     local_batch['level_noisy'] = scheduler.step(pred['next_state_level'], t, 
                #                                                 local_batch['level_noisy'], 
                #                                                 generator=generator).prev_sample

                
              #  local_batch['surface_noisy'] = remove_outliers(local_batch['surface_noisy'])
               # steps.append(local_batch['surface_noisy'])
                steps.append(original_sample)
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
            sample = dict(next_state_surface=local_batch['surface_noisy'],state_surface=local_batch['state_surface']) 
            
        if denormalize:
            #sample,batch = self.dataset.denormalize(sample, batch)
            pass
        #denorm_sample = {k:v.detach() for k, v in denorm_sample.items()}
        return sample,steps

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
    # def test_step(self, batch, batch_nb):
    #     scratch = os.environ['SCRATCH']
    #     run_id = '20e12882'
    #     file_name = 'epoch=45.ckpt'
    #     checkpoint_path = f'{scratch}/checkpoint_{run_id}/{file_name}'

    #     self.init_from_ckpt(checkpoint_path)
    #     device = batch['state_surface'].device
    #     x = np.stack(self.dataset.timestamps)[:,2]
    #     indices = np.stack(self.dataset.timestamps)[np.where((pd.to_datetime(x).year == 2001) & (pd.to_datetime(x).month == 1),True,False)][:,[0,1]]
    #     # 
    #     # 
    #     # checkpoint_path = torch.load(f'{scratch}/checkpoint_{run_id}/{file_name}',map_location=torch.device('cuda'))
    #     # model.load_state_dict(checkpoint_path['state_dict'])
    #     # # trainer.test(model, val_dataloader)

    #     inv_map = {v: k for k, v in self.dataset.id2pt.items()}
    #     index = inv_map[(indices[0][0],indices[0][1])]

    #     batch = self.dataset.__getitem__(index)
    #     # batch['state_surface'] = batch['state_surface'].to(device)
    #     # batch['prev_state_surface'] = batch['prev_state_surface'].to(device)
    #     # batch['forcings'] = batch['forcings'].to(device)
    #     batch['state_constant'] = torch.tensor(batch['state_constant'])
    #     batch['time'] = [batch['time']]
    #     batch['next_time'] = [batch['next_time']]
    #     for k, v in batch.items():
    #         print(k)
    #         if(k != 'time' and k != 'next_time'):
    #             batch[k] = batch[k].to(device)
        

    #     print(batch['time'])
    #     batch = {k: v.unsqueeze(0) if (k != 'time') and (k != 'next_time') else v for k, v in batch.items()}        
    #     for i in range(3):
    #         print(i,'ensemble member')
    #         output = self.sample_rollout(batch, lead_time_months=60,seed = i)
    #         with open(f'{i}_rollout_v_predictions_30year_ssp_585_{run_id}_50_inference_steps.pkl','wb') as f:
    #             pickle.dump(output,f)
