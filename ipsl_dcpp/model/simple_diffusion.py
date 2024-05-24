
import lightning.pytorch as pl
import diffusers
import torch
from ipsl_dcpp.model.pangu import TimestepEmbedder
from ipsl_dcpp.evaluation.metrics import EnsembleMetrics
import datetime
import numpy as np
import os
import psutil
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
        num_ensemble_members
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
    def mylog(self, dct={}, **kwargs):
        mode = 'train_' if self.training else 'val_'
        dct.update(kwargs)
        for k, v in dct.items():
            self.log(mode+k, v, prog_bar=True,sync_dist=True)

    def forward(self, batch, timesteps, sel=1):

        device = batch['state_surface'].device
        bs = batch['state_surface'].shape[0]
      #  print(sel.shape)
 #       print(batch['surface_noisy'].shape)
 #       print(batch['state_surface'].shape)
 #       print(batch['state_constant'].shape)      
       # batch['surface_noisy'] = batch['surface_noisy'].squeeze(1)
        batch['state_surface'] = torch.cat([batch['state_surface'],batch['prev_state_surface'], 
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

        # Get the target for loss
        #target_surface = batch['next_state_surface']
       # target_level = batch['next_state_level']
        
        # create uncond
        # sel = (torch.rand((bs,), device=device))
        pred = self.forward(batch, timesteps)
        # compute loss
       # batch['next_state_surface'] = target_surface
        #batch['next_state_level'] = target_level

        _, _, loss = self.loss(pred, batch)
        loss = loss.mean()
        self.mylog(loss=loss)

        return loss

    def loss(self, pred, batch, lat_coeffs=None):
        if lat_coeffs is None:
            lat_coeffs = self.dataset.lat_coeffs_equi
    # surface_coeffs = pangu_surface_coeffs
        device = batch['next_state_surface'].device
      #  print(pred['next_state_surface'].squeeze().shape)
      #  print(batch['next_state_surface'].squeeze().shape)
    #    print(pred['next_state_surface'][0,0,100])
   #     print(batch['next_state_surface'][0,0,0,100])
        mse_surface = (pred['next_state_surface'].squeeze() - batch['next_state_surface'].squeeze()).pow(2)

        mse_surface = mse_surface.mul(lat_coeffs.to(device)) # latitude coeffs
    #   mse_surface_w = mse_surface.mul(surface_coeffs.to(device))
    
    # mse_level = (pred['next_state_level'] - batch['next_state_level']).pow(2)
    # if mse_level.shape[-2] == 128:
    #     mse_level = mse_level[..., 4:-4, 8:-8]
    # mse_level = mse_level.mul(lat_coeffs.to(device))
    # mse_level_w = mse_level.mul(level_coeffs.to(device))

        #if self.selected_vars:
        #    coeffs = 0.01*torch.ones(1, 5, 13, 1, 1).to(device)
        #    coeffs[:, 0, 7] = 1
        #    mse_level_w *= coeffs
    
    # nvar = (surface_coeffs.sum().item() + 5)
        
    #   loss = (mse_surface_w.sum(1).mean((-3, -2, -1)) + 
    #           mse_level_w.sum(1).mean((-3, -2, -1)))/nvar
        loss = mse_surface.sum(1).mean((-3, -2, -1))
        return mse_surface, None, loss

    def sample_rollout(self, batch, *args, lead_time_months, **kwargs):
        history = dict(state_surface=[])
        next_time = batch['next_time']    
       # print(next_time)
        # get_year_index = np.vectorize(lambda x: int(x.split('-')[0]) - 1960)

        # cur_year_index = torch.Tensor(get_year_index(next_time))
        cur_year_index = int(next_time[0].split('-')[0]) - 1960
        cur_month_index = int(next_time[0].split('-')[-1]) - 1

        #print(cur_year_index)
        #print(self.dataset.atmos_forcings[:,cur_year_index].shape)
       #print(self.dataset.atmos_forcings.shape)
        inc_time_vec = np.vectorize(inc_time)
     #   local_batch = {k:v for k, v in batch.items() if not k.startswith('next')} # ensure no data leakage
        for i in range(lead_time_months):
            sample,batch = self.sample(batch, denormalize=False,num_inference_steps=self.num_inference_steps,scheduler=self.scheduler)
            history['state_surface'].append(sample['next_state_surface'])
           # history['state_level'].append(denorm_sample['next_state_level'])
            # make new batch starting from denorm_sample
            # print(denorm_sample['next_state_surface'].shape)
            # print(local_batch['state_surface'].shape)
            # print(self.dataset.surface_delta_stds.shape)
            # print(state_surface[0].shape)
           # print(cur_year_index)
           # print(self.dataset.atmos_forcings.shape)
           # print(self.dataset.atmos_forcings[:,cur_year_index].shape)
            print(len(sample))
            print(sample)
           #only for delta runs
            state_surface=sample['next_state_surface']*np.expand_dims(self.dataset.surface_delta_stds,0) + batch['state_surface'],
            print(state_surface)
            batch = dict(state_surface=state_surface[0],
                         prev_state_surface=batch['state_surface'],
                               #state_level=denorm_sample['next_state_level'],
                               next_state_surface=batch['next_state_surface'],
                               time=next_time,
                               state_constant=batch['state_constant'],
                                 forcings=torch.Tensor(self.dataset.atmos_forcings[:,cur_year_index]).unsqueeze(0), #I have no idea why i need to unsqueeze here does lightning add a dimenions with the batch? who knows
                                 state_depth=torch.empty(0),
                                 solar_forcings=torch.Tensor(self.dataset.solar_forcings[cur_year_index,cur_month_index]).unsqueeze(0),
                               next_time=inc_time_vec(next_time))
       #     print(local_batch['forcings'].shape)
        # it should return the history of generated states !
        return history
        
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

    def sample(self, batch, 
                scheduler,
                num_inference_steps,
                seed=0,
                denormalize=True,
                cf_guidance=None,
                ):
        if cf_guidance is None:
            cf_guidance = 1
        if scheduler == 'ddpm':
            scheduler = self.noise_scheduler

        scheduler.set_timesteps(num_inference_steps)

        local_batch = {k:v for k, v in batch.items() if not k.startswith('next')} # ensure no data leakage
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)

        surface_noise = torch.randn(local_batch['state_surface'].size(), generator=generator)

        local_batch['surface_noisy'] = surface_noise.to(self.device)
      #  steps = [] # this is the memory leak
        with torch.no_grad():
            for t in scheduler.timesteps:
                # print(t)
                # 1. predict noise model_output
                pred = self.forward(local_batch, torch.tensor([t]).to(self.device))
                # if cf_guidance > 1:
                #     uncond_pred = self.forward(local_batch, torch.tensor([t]).to(self.device), 0)
                #     # compute epsilon from uncond_pred 
                #     pred = dict(next_state_surface=pred['next_state_surface']*(1+cf_guidance) - uncond_pred['next_state_surface']*cf_guidance,
                #                 next_state_level=pred['next_state_level']*(1+cf_guidance) - uncond_pred['next_state_level']*cf_guidance)
                
                local_batch['surface_noisy'] = scheduler.step(pred['next_state_surface'], t, 
                                                            local_batch['surface_noisy'], 
                                                            generator=generator).prev_sample
               # steps.append(local_batch['surface_noisy'])
                # print(local_batch['state_surface'].shape,'local batch')
                # a hack to readjust what is modified in the forward loop
                local_batch['state_surface'] = local_batch['state_surface'][:,:9]
            sample = dict(next_state_surface=local_batch['surface_noisy'])
            
        if denormalize:
            sample,batch = self.dataset.denormalize(sample, batch)

        #denorm_sample = {k:v.detach() for k, v in denorm_sample.items()}
        return sample,batch

    def validation_step(self, batch, batch_nb):
            # for the validation, we make some generations and log them 
            samples = [self.sample(batch, num_inference_steps=self.num_inference_steps, 
                                    denormalize=True, 
                                    scheduler='ddpm', 
                                    seed = j)
                                for j in range(self.num_members)]
            
            # denorm_samples = [self.dataset.denormalize(s, batch) for s in samples]
            denorm_batch = samples[0][1]
            denorm_samples = [x[0] for x in samples]
            print(psutil.Process(os.getpid()).memory_info()[0]/(2.**30),'validation')

            #denorm_sample1, denorm_batch = self.dataset.denormalize(sample1, batch)
            #denorm_sample2, _ = self.dataset.denormalize(sample2, batch)

            self.metrics.update(denorm_batch, denorm_samples)
            samples = None
          #  self.classifier_score.update(denorm_batch, denorm_samples)
            # avg = dict(next_state_level=(sample1['next_state_level']+sample2['next_state_level'])/2,
            #             next_state_surface = (sample1['next_state_surface']+sample2['next_state_surface'])/2)
            
            # err_level = mse(batch['next_state_level'] - avg['next_state_level']).mean(0)
            # err_surface = mse(batch['next_state_surface'] - avg['next_state_surface']).mean(0)
            
            # var_level = mse(torch.cat([sample1['next_state_level'],
            #                        sample2['next_state_level']], 0).var(0).sqrt()).mean(0)
            # var_surface = mse(torch.cat([sample1['next_state_surface'],
            #                           sample2['next_state_surface']], 0).var(0).sqrt()).mean(0)

            # n_members = sample1['next_state_surface'].shape[0] + sample2['next_state_surface'].shape[0]

            # skr_level = (1 + 1/n_members)**.5 * var_level.sqrt()/err_level.sqrt()
            # skr_surface = (1 + 1/n_members)**.5 * var_surface.sqrt()/err_surface.sqrt()

            # # also check cross-seed variance ?
            # avg_seed = dict(next_state_level=(sample1['next_state_level'][0]+sample1['next_state_level'][1])/2,
            #             next_state_surface = (sample1['next_state_surface'][0]+sample1['next_state_surface'][1])/2)

            # csvar_level = mse(torch.stack([sample1['next_state_level'][0],
            #                             sample2['next_state_level'][0]]).var(0).sqrt())[0]
            # csvar_surface = mse(torch.stack([sample1['next_state_surface'][0],
            #                                 sample2['next_state_surface'][0]]).var(0).sqrt())[0]
            # n_members = 2
            # csr_level = (1 + 1/n_members)**.5 * csvar_level.sqrt()/err_level.sqrt()
            # csr_surface = (1 + 1/n_members)**.5 * csvar_surface.sqrt()/err_surface.sqrt()
                        
            # self.mylog(skr_z500=skr_level[0, 7])
            # self.mylog(skr_t2m=skr_surface[2, 0])

            # self.mylog(csr_z500=csr_level[0, 7])
            # self.mylog(csr_t2m=csr_surface[2, 0])
            # sample1, sample2 = samples[:2]
            # if hasattr(self.logger, 'log_image'):
            #     for i in range(batch['next_state_surface'].shape[0]):
            #         t2m_images = [batch['next_state_surface'][i, 2, 0], sample1['next_state_surface'][i, 2, 0], sample2['next_state_surface'][i, 2, 0]]
            #         z500_images = [batch['next_state_level'][i, 0, 7], sample1['next_state_level'][i, 0, 7], sample2['next_state_level'][i, 0, 7]]
            #         m_t2m = min([x.min() for x in t2m_images])
            #         M_t2m = max([x.max() for x in t2m_images])
            #         M_t2m = max([M_t2m, -m_t2m])
            #         m_z500 = min([x.min() for x in z500_images])
            #         M_z500 = max([x.max() for x in z500_images])
            #         M_z500 = max([M_z500, -m_z500])

            #         t2m_images = [bw_to_bwr(x, m=-M_t2m, M=M_t2m) for x in t2m_images]
            #         z500_images = [bw_to_bwr(x, m=-M_z500, M=M_z500) for x in z500_images]

            #         self.logger.log_image(key='t2m samples', images=t2m_images, caption = ['gt '+batch['time'][i], 'sample1', 'sample2'], step=self.global_step)
            #         self.logger.log_image(key='z500 samples', images=z500_images, caption = ['gt '+batch['time'][i], 'sample1', 'sample2'], step=self.global_step)

            #         # lets only log once for now
            #         break

            return None
            #return self.training_step(batch, batch_nb)


    def on_validation_epoch_end(self):
        for metric in [self.metrics]:
            out = metric.compute()
            self.log_dict(out)# dont put on_epoch = True here
            print(out)
            metric.reset()

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)
