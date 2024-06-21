
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


class SimpleDiffusion(pl.LightningModule):
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
        
        device = batch['state_surface'].device
        bs = batch['state_surface'].shape[0]
        #print(sel.shape)
        # print(batch['surface_noisy'].shape)
        # print(batch['state_surface'].shape)
        # print(batch['state_constant'].shape)
        # print(batch['prev_state_surface'].shape)
       # batch['surface_noisy'] = batch['surface_noisy'].squeeze(1)
        batch['state_surface'] = torch.cat([batch['state_surface']*sel,batch['prev_state_surface']*sel, 
                                   batch['surface_noisy'],batch['state_constant']], dim=1)
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
        # sample timesteps
        device = batch['state_surface'].device
        bs = batch['state_surface'].shape[0]

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

        surface_noise = torch.randn_like(batch['next_state_surface'])
        #level_noise = torch.randn_like(batch['next_state_level'])
        batch['surface_noisy'] = self.noise_scheduler.add_noise(batch['next_state_surface'], surface_noise, timesteps)
        #batch['level_noisy'] = self.noise_scheduler.add_noise(batch['next_state_level'], level_noise, timesteps)

        batch['surface_noisy'] = self.noise_scheduler.scale_model_input(batch['surface_noisy'])
        #batch['level_noisy'] = self.noise_scheduler.scale_model_input(batch['level_noisy'])
        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target_surface = surface_noise
        #    target_level = level_noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target_surface = self.noise_scheduler.get_velocity(batch['next_state_surface'], surface_noise, timesteps)
     #       target_level = self.noise_scheduler.get_velocity(batch['next_state_level'], level_noise, timesteps)

        elif self.noise_scheduler.config.prediction_type == "sample":
            target_surface = batch['next_state_surface']
      #      target_level = batch['next_state_level']

        
        # create uncond
        sel = (torch.rand((bs,), device=device) > self.p_uncond)
        pred = self.forward(batch, timesteps,sel[:,None,None,None])
        # compute loss
        batch['next_state_surface'] = target_surface
    #    batch['next_state_level'] = target_level

        _, _, loss = self.loss(pred, batch)
        loss = loss.mean()
        self.mylog(loss=loss)

        return loss

    def loss(self, pred, batch, lat_coeffs=None):
        if lat_coeffs is None:
            lat_coeffs = self.dataset.lat_coeffs_equi
        device = batch['next_state_surface'].device
        mse_surface = (pred['next_state_surface'].squeeze() - batch['next_state_surface'].squeeze()).pow(2)
        mse_surface = mse_surface.mul(lat_coeffs.to(device)) # latitude coeffs
        loss = mse_surface.sum(1).mean((-3, -2, -1))
        return mse_surface, None, loss

    def sample_rollout(self, batch, *args, lead_time_months, seed,**kwargs):
        device = batch['state_surface'].device
        history = dict(state_surface=[])
        next_time = batch['next_time']    
        cur_year_index = int(next_time[0].split('-')[0]) - 1960
        cur_month_index = int(next_time[0].split('-')[-1]) - 1
        inc_time_vec = np.vectorize(inc_time)


        for i in range(lead_time_months):
            start = time.time()

            print(i,'lead_time')
            sample = self.sample(batch, denormalize=False,num_inference_steps=self.num_inference_steps,scheduler='ddpm',seed=seed)
#            print(len(sample))
#            print(sample)
            #only for delta runs
            print(sample)
            next_time = batch['next_time']    
            cur_year_index = int(next_time[0].split('-')[0]) - 1960
            cur_month_index = int(next_time[0].split('-')[-1]) - 1
            # print(sample['next_state_surface'].shape)
            # print(batch['state_surface'].shape)
            # print(self.dataset.surface_delta_stds.shape)
            new_state_surface=sample['next_state_surface'].to(device)*torch.unsqueeze(self.dataset.surface_delta_stds,0).to(device) + batch['state_surface'].to(device)
          #  prev_state_surface=sample['state_surface'].to(device)*torch.unsqueeze(self.dataset.surface_delta_stds,0).to(device) + batch['state_surface'].to(device)
            
            history['state_surface'].append(new_state_surface)

#            print(state_surface)
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
            sched_cfg = self.noise_scheduler.config
            scheduler = DDIMScheduler(num_train_timesteps=sched_cfg.num_train_timesteps,
                                                       beta_schedule=sched_cfg.beta_schedule,
                                                       prediction_type=sched_cfg.prediction_type,
                                                       beta_start=sched_cfg.beta_start,
                                                       beta_end=sched_cfg.beta_end,
                                                       clip_sample=False,
                                                       clip_sample_range=1e6)
            

        scheduler.set_timesteps(num_inference_steps)

        local_batch = {k:v for k, v in batch.items() if not k.startswith('next')} # ensure no data leakage
        generator = torch.Generator(device=batch['state_surface'].device)
        generator.manual_seed(seed)
    
        surface_noise = torch.randn(local_batch['state_surface'].size(), generator=generator)
        steps = []
        local_batch['surface_noisy'] = surface_noise.to(self.device)
        with torch.no_grad():
            for t in scheduler.timesteps:
              #  prev_t = scheduler.previous_timestep(t)

            #    alpha_prod_t = scheduler.alphas_cumprod[t]
              #  alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.one
             #   beta_prod_t = 1 - alpha_prod_t
            #    beta_prod_t_prev = 1 - alpha_prod_t_prev
            #    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
             #   current_beta_t = 1 - current_alpha_t
                print(t)
                print(local_batch['surface_noisy'][0,0,0,:],'next time step')

                #sel = (torch.rand((1,), device='cpu') > self.p_uncond)
                pred = self.forward(local_batch, torch.tensor([t]).to(self.device))
                #scheduler step finds the actual output (in this case, the delta) where next_state_surface is the velocity predicted
                #uses https://github.com/huggingface/diffusers/blob/668e34c6e0019b29c887bcc401c143a7d088cb25/src/diffusers/schedulers/scheduling_ddpm.py#L525
                local_batch['surface_noisy'] = scheduler.step(pred['next_state_surface'], t, 
                                                            local_batch['surface_noisy'], 
                                                            generator=generator).prev_sample
                #(alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
               # pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output

               # local_batch['surface_noisy'] = (((alpha_prod_t**0.5) * local_batch['surface_noisy']) - vel) /(beta_prod_t**0.5)
                #due to the way the model is set up, the state_surface is modified in the forward loop
                steps.append(local_batch['state_surface'])
                local_batch['state_surface'] = local_batch['state_surface'][:,:9]
            
            sample = dict(next_state_surface=local_batch['surface_noisy'],state_surface=local_batch['state_surface'])
            
        if denormalize:
            sample,batch = self.dataset.denormalize(sample, batch)

        #denorm_sample = {k:v.detach() for k, v in denorm_sample.items()}
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
