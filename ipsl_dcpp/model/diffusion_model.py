import torch
import diffusers
from diffusers.training_utils import compute_snr

from evaluation.metrics import EnsembleMetrics


from .forecast import ForecastModule, pangu_surface_coeffs, level_coeffs #, lat_coeffs_equi

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

class DiffusionModule(ForecastModule):
    '''
    this module is for learning the diffusion stuff
    '''
    def __init__(self, 
                 backbone,
                 dataset,
                 cond_dim=32, 
                 num_train_timesteps=1000,
                 scheduler='ddpm',
                 prediction_type="v_prediction",
                 beta_schedule='squaredcos_cap_v2',
                 beta_start=0.0001,
                 beta_end=0.012,
                 snr_gamma=None,
                 flow_matching=False,
                 conditional=False,
                 div_noisy_input=False,
                 uncond_proba=0.,
                 two_branches=False, # I think we will remove this
                 ckpt_path=None,
                 ft_lr=1e-5,
                 sub_pred=False,
                 snr_reweighting=False, # this is useful for sample prediction.
                 selected_vars=False,
                 num_inference_steps=50,
                 cf_guidance=1,
                 num_members=4,
                 eval_classifier=True,
                 debug=False,
                 **kwargs):
        super().__init__(backbone, dataset, **kwargs)
        self.__dict__.update(locals())
        # cond_dim should be given as arg to the backbone
        from backbones.dit import TimestepEmbedder
        self.month_embedder = TimestepEmbedder(cond_dim)
        self.hour_embedder = TimestepEmbedder(cond_dim)
        self.timestep_embedder = TimestepEmbedder(cond_dim)
        self.noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=num_train_timesteps,
                                                    beta_schedule=beta_schedule,
                                                    beta_start=beta_start,
                                                    beta_end=beta_end,
                                                    prediction_type=prediction_type,
                                                    clip_sample=False,
                                                    clip_sample_range=1e6,
                                                    rescale_betas_zero_snr=True,
                                                    )
        self.metrics = EnsembleMetrics(dataset=dataset)

    def forward(self, batch, timesteps, sel=1):
        device = batch['state_surface'].device
        bs = batch['state_surface'].shape[0]
        df = 100 if self.div_noisy_input else 1
        if self.two_branches:
            input_surface = torch.cat([
                                    batch['state_constant'], 
                                    batch['pred_state_surface']*sel], dim=1)
            input_level = batch['pred_state_level']*sel

            noisy_surface = torch.cat([batch['state_constant'], 
                                    batch['surface_noisy']], dim=1).squeeze(-3)
            
            noisy_level = batch['level_noisy']

        elif self.conditional == 'state_only':
            input_surface = torch.cat([batch['state_surface']*sel, 
                                    batch['state_constant'], 
                                    batch['surface_noisy']/df], dim=1)
            input_level = torch.cat([batch['state_level']*sel,
                                    batch['level_noisy']/df], dim=1) 

        elif self.conditional == 'pred_only':
            input_surface = torch.cat([batch['pred_state_surface']*sel, 
                                    batch['state_constant'], 
                                    batch['surface_noisy']/df], dim=1)
            input_level = torch.cat([batch['pred_state_level']*sel,
                                    batch['level_noisy']/df], dim=1) 
        
        elif self.conditional == 'noisy_branch' or self.conditional == 'noisy_branch_cat':
            input_surface = torch.cat([
                                    batch['state_surface']*sel,
                                    batch['state_constant'], 
                                    ], dim=1)
            input_level = batch['state_level']*sel

        elif self.conditional == 'noisy_branch_pred':
            input_surface = torch.cat([
                                    batch['pred_state_surface']*sel,
                                    batch['state_constant'], 
                                    ], dim=1)
            input_level = batch['pred_state_level']*sel
        elif self.conditional == 'all':
            input_surface = torch.cat([batch['state_surface']*sel, 
                                    batch['state_constant'], 
                                    batch['pred_state_surface']*sel,
                                    batch['surface_noisy']/df], dim=1)
            input_level = torch.cat([batch['state_level']*sel,
                                    batch['pred_state_level']*sel,
                                    batch['level_noisy']/df], dim=1)
        else:
            input_surface = torch.cat([
                                    batch['state_constant'], 
                                    batch['surface_noisy']/df], dim=1)
            input_level = batch['level_noisy']    
        
        input_surface = input_surface.squeeze(-3)
        
        month = torch.tensor([int(x[5:7]) for x in batch['time']]).to(device)
        month_emb = self.month_embedder(month)
        hour = torch.tensor([int(x[-5:-3]) for x in batch['time']]).to(device)
        hour_emb = self.hour_embedder(hour)

        timestep_emb = self.timestep_embedder(timesteps)

        cond_emb = month_emb + hour_emb + timestep_emb

        if self.two_branches:
            out = self.backbone(input_surface, input_level, 
                                noisy_surface, noisy_level, cond_emb)
        elif self.conditional in ('noisy_branch', 'noisy_branch_pred'):
            # no pred
            noisy_surface = torch.cat([batch['state_constant'], 
                                    batch['surface_noisy']], dim=1).squeeze(-3)
            
            noisy_level = batch['level_noisy']
            parrallel_input = self.noisy_branch(noisy_surface, noisy_level, cond_emb)

            out = self.backbone(input_surface, input_level, cond_emb, parrallel_input=0.1*parrallel_input)


        elif self.conditional == 'noisy_branch_cat':

            noisy_surface = torch.cat([batch['state_constant'], 
                                    batch['pred_state_surface'],
                                    batch['surface_noisy']], dim=1).squeeze(-3)
            
            noisy_level = torch.cat([batch['pred_state_level'],
                                     batch['level_noisy']], dim=1)

            parrallel_input = self.noisy_branch(noisy_surface, noisy_level, cond_emb)
            out = self.backbone(input_surface, input_level, cond_emb, parrallel_input=0.1*parrallel_input)

        else:
            out = self.backbone(input_surface, input_level, cond_emb)
        
        if self.sub_pred:
            out['next_state_surface'] = out['next_state_surface'] - batch['pred_state_surface']
            out['next_state_level'] = out['next_state_level'] - batch['pred_state_level']

        if self.noise_scheduler.config.prediction_type == "v_prediction" and self.ckpt_path is not None:
            # estimate input noise
            alpha = self.noise_scheduler.alphas_cumprod.to(self.device)[timesteps][:, None, None, None, None]

            est_surface_noise = (batch['surface_noisy'] - out['next_state_surface']*alpha.sqrt())/(1-alpha).sqrt().add(1e-6)
            est_level_noise = (batch['level_noisy'] - out['next_state_level']*alpha.sqrt())/(1-alpha).sqrt().add(1e-6)

            out = dict(next_state_surface=self.noise_scheduler.get_velocity(
                             out['next_state_surface'], est_surface_noise, timesteps),
                             next_state_level=self.noise_scheduler.get_velocity(
                             out['next_state_level'], est_level_noise, timesteps))
        return out
    
    def training_step(self, batch, batch_nb):
        # sample timesteps
        device = batch['state_surface'].device
        bs = batch['state_surface'].shape[0]

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

        surface_noise = torch.randn_like(batch['next_state_surface'])
        level_noise = torch.randn_like(batch['next_state_level'])

        if not self.flow_matching:
            batch['surface_noisy'] = self.noise_scheduler.add_noise(batch['next_state_surface'], surface_noise, timesteps)
            batch['level_noisy'] = self.noise_scheduler.add_noise(batch['next_state_level'], level_noise, timesteps)

            batch['surface_noisy'] = self.noise_scheduler.scale_model_input(batch['surface_noisy'])
            batch['level_noisy'] = self.noise_scheduler.scale_model_input(batch['level_noisy'])

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target_surface = surface_noise
                target_level = level_noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target_surface = self.noise_scheduler.get_velocity(batch['next_state_surface'], surface_noise, timesteps)
                target_level = self.noise_scheduler.get_velocity(batch['next_state_level'], level_noise, timesteps)

            elif self.noise_scheduler.config.prediction_type == "sample":
                target_surface = batch['next_state_surface']
                target_level = batch['next_state_level']
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
            
        elif self.flow_matching:
            coeff = timesteps / self.noise_scheduler.config.num_train_timesteps
            coeff = coeff[:, None, None, None, None]
            batch['surface_noisy'] = coeff * batch['state_surface'] + (1 - coeff)*surface_noise
            batch['level_noisy'] = coeff * batch['state_level'] + (1 - coeff)*level_noise

            target_surface = batch['next_state_surface'] - surface_noise
            target_level = batch['next_state_level'] - level_noise
  
        # create uncond
        sel = (torch.rand((bs,), device=device) > self.uncond_proba)
        pred = self.forward(batch, timesteps, sel.float()[:, None, None, None, None])
        #pred_cond = self.forward(batch, timesteps, 1)

        #if self.uncond_proba > 0:
        #    pred_uncond = self.forward(batch, timesteps, 0)

        # compute loss
        batch['next_state_surface'] = target_surface
        batch['next_state_level'] = target_level

        _, _, per_sample_loss_cond = self.loss(pred, batch)


        #if self.uncond_proba > 0:
        #    _, _, per_sample_loss_uncond = self.loss(pred_uncond, batch)

        # reweight loss with min snr gamma
        # TODO: not compatible for now
        if self.snr_gamma:
            snr = compute_snr(self.noise_scheduler, timesteps)
            base_weight = torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] 
            if self.noise_scheduler.config.prediction_type == 'epsilon':
                base_weight = base_weight / snr

            if self.noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective needs to be floored to an SNR weight of one.
                mse_loss_weights = base_weight + 1
            else:
                # Epsilon and sample both use the same loss weights.
                mse_loss_weights = base_weight
                
            loss = (per_sample_loss_cond * mse_loss_weights)
        else:
            loss = per_sample_loss_cond
            # if self.uncond_proba > 0:
            #     loss = (per_sample_loss_cond + per_sample_loss_uncond)/2
            # else:
            #     loss = per_sample_loss_cond
            
        #if sel.max() > 0:
        self.mylog(cond_loss=per_sample_loss_cond.mean())
        #if (~sel).max() > 0:
        #if self.uncond_proba > 0:
        #    self.mylog(uncond_loss=per_sample_loss_uncond.mean())
        #final loss
        loss = loss.mean()
        self.mylog(loss=loss)

        return loss
    
    def loss(self, pred, batch, lat_coeffs=None):
        if lat_coeffs is None:
            lat_coeffs = self.dataset.lat_coeffs_equi
        surface_coeffs = pangu_surface_coeffs
        device = batch['next_state_level'].device
        mse_surface = (pred['next_state_surface'] - batch['next_state_surface']).pow(2)
        if mse_surface.shape[-2] == 128:
            mse_surface = mse_surface[..., 4:-4, 8:-8]
        mse_surface = mse_surface.mul(lat_coeffs.to(device)) # latitude coeffs
        mse_surface_w = mse_surface.mul(surface_coeffs.to(device))
    
        mse_level = (pred['next_state_level'] - batch['next_state_level']).pow(2)
        if mse_level.shape[-2] == 128:
            mse_level = mse_level[..., 4:-4, 8:-8]
        mse_level = mse_level.mul(lat_coeffs.to(device))
        mse_level_w = mse_level.mul(level_coeffs.to(device))

        if self.selected_vars:
            coeffs = 0.01*torch.ones(1, 5, 13, 1, 1).to(device)
            coeffs[:, 0, 7] = 1
            mse_level_w *= coeffs
    
        nvar = (surface_coeffs.sum().item() + 5)
        
        loss = (mse_surface_w.sum(1).mean((-3, -2, -1)) + 
                mse_level_w.sum(1).mean((-3, -2, -1)))/nvar
        
        return mse_surface, mse_level, loss

    def sample(self, batch, 
               scheduler='ddpm',
               num_inference_steps=50,
               seed=0,
               denormalize=True,
               cf_guidance=None,
               ):
        if cf_guidance is None:
            cf_guidance = int(self.cf_guidance)
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
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)

        surface_noise = torch.randn(local_batch['state_surface'].size(), generator=generator)
        level_noise = torch.randn(local_batch['state_level'].size(), generator=generator)

        local_batch['surface_noisy'] = surface_noise.to(self.device)
        local_batch['level_noisy'] = level_noise.to(self.device)

        with torch.no_grad():
            for t in scheduler.timesteps:
                # 1. predict noise model_output
                pred = self.forward(local_batch, torch.tensor([t]).to(self.device), (1 if cf_guidance > 0 else 0))

                if cf_guidance > 1:
                    uncond_pred = self.forward(local_batch, torch.tensor([t]).to(self.device), 0)
                    # compute epsilon from uncond_pred 
                    pred = dict(next_state_surface=pred['next_state_surface']*(1+cf_guidance) - uncond_pred['next_state_surface']*cf_guidance,
                                next_state_level=pred['next_state_level']*(1+cf_guidance) - uncond_pred['next_state_level']*cf_guidance)
                
                local_batch['surface_noisy'] = scheduler.step(pred['next_state_surface'], t, 
                                                            local_batch['surface_noisy'], 
                                                            generator=generator).prev_sample

                local_batch['level_noisy'] = scheduler.step(pred['next_state_level'], t, 
                                                            local_batch['level_noisy'], 
                                                            generator=generator).prev_sample
                
            sample = dict(next_state_surface=local_batch['surface_noisy'],
                        next_state_level=local_batch['level_noisy'])
            
        denorm_sample = sample
        if denormalize:
            denorm_sample = self.dataset.denormalize(sample, batch)

        denorm_sample = {k:v.detach() for k, v in denorm_sample.items()}
        return denorm_sample

    def sample_rollout(self, batch, *args, lead_time_days=1, **kwargs):
        history = dict(state_surface=[],state_level=[])
        local_batch = {k:v for k, v in batch.items() if not k.startswith('next')} # ensure no data leakage
        for i in range(lead_time_days):
            denorm_sample = self.sample(local_batch, *args, **kwargs)
            history['state_surface'].append(denorm_sample['next_state_surface'])
            history['state_level'].append(denorm_sample['next_state_level'])
            # make new batch starting from denorm_sample
            local_batch = dict(state_surface=denorm_sample['next_state_surface'],
                               state_level=denorm_sample['next_state_level'],
                               time=batch['next_time'])
            # TODO: pb: this needs the base inference model !!

        # it should return the history of generated states !
        return None


    def validation_step(self, batch, batch_nb):
        # for the validation, we make some generations and log them 
        samples = [self.sample(batch, num_inference_steps=self.num_inference_steps, 
                                denormalize=False, 
                                scheduler='ddim', 
                                seed = j)
                            for j in range(self.num_members)]
        
        denorm_samples = [self.dataset.denormalize(s, batch) for s in samples]
        denorm_batch = self.dataset.denormalize(batch)
        #denorm_sample1, denorm_batch = self.dataset.denormalize(sample1, batch)
        #denorm_sample2, _ = self.dataset.denormalize(sample2, batch)

        self.metrics.update(denorm_batch, denorm_samples)
        self.classifier_score.update(denorm_batch, denorm_samples)
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
        sample1, sample2 = samples[:2]
        if hasattr(self.logger, 'log_image'):
            for i in range(batch['next_state_surface'].shape[0]):
                t2m_images = [batch['next_state_surface'][i, 2, 0], sample1['next_state_surface'][i, 2, 0], sample2['next_state_surface'][i, 2, 0]]
                z500_images = [batch['next_state_level'][i, 0, 7], sample1['next_state_level'][i, 0, 7], sample2['next_state_level'][i, 0, 7]]
                m_t2m = min([x.min() for x in t2m_images])
                M_t2m = max([x.max() for x in t2m_images])
                M_t2m = max([M_t2m, -m_t2m])
                m_z500 = min([x.min() for x in z500_images])
                M_z500 = max([x.max() for x in z500_images])
                M_z500 = max([M_z500, -m_z500])

                t2m_images = [bw_to_bwr(x, m=-M_t2m, M=M_t2m) for x in t2m_images]
                z500_images = [bw_to_bwr(x, m=-M_z500, M=M_z500) for x in z500_images]

                self.logger.log_image(key='t2m samples', images=t2m_images, caption = ['gt '+batch['time'][i], 'sample1', 'sample2'], step=self.global_step)
                self.logger.log_image(key='z500 samples', images=z500_images, caption = ['gt '+batch['time'][i], 'sample1', 'sample2'], step=self.global_step)

                # lets only log once for now
                break

        return None
        #return self.training_step(batch, batch_nb)
    
    def on_validation_epoch_end(self):
        for metric in [self.metrics, self.classifier_score]:
            out = metric.compute()
            self.log_dict(out)# dont put on_epoch = True here
            print(out)
            metric.reset()

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith('classifier_score'):
                del checkpoint['state_dict'][k]
    
    def on_load_checkpoint(self, checkpoint):
        # add classifier weights 
            checkpoint['state_dict'].update({'classifier_score.'+k:v for k, v in self.classifier_score.state_dict().items()})

    
    def configure_optimizers(self):
        print('configure optimizers')
        if self.ckpt_path is not None:
            opt = torch.optim.AdamW([{'params': self.backbone.parameters(), 'lr': self.ft_lr}, # finetune
                                 {'params': self.noisy_branch.parameters()},
                                 {'params': self.month_embedder.parameters()},
                                 {'params': self.hour_embedder.parameters()},
                                 {'params': self.timestep_embedder.parameters()}], 
                                lr=self.lr, betas=self.betas, 
                                weight_decay=self.weight_decay)
        else:
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
