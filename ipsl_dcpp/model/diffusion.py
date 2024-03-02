import pytorch_lightning as pl
import torch.nn as nn
import torch
import diffusers
from pathlib import Path
from diffusers.training_utils import compute_snr


from .forecast import ForecastModule, pressure_levels, surface_coeffs, level_coeffs, lat_coeffs_equi

class DiffusionModule(ForecastModule):
    '''
    this module is for learning the diffusion stuff
    '''
    def __init__(self, *args, 
                 cond_dim=32, 
                 num_train_timesteps=1000,
                 prediction_type="v_prediction",
                 beta_schedule='squaredcos_cap_v2',
                 snr_gamma=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.snr_gamma = snr_gamma
        # cond_dim should be given as arg to the backbone
        from backbones.dit import TimestepEmbedder
        self.month_embedder = TimestepEmbedder(cond_dim)
        self.hour_embedder = TimestepEmbedder(cond_dim)
        self.timestep_embedder = TimestepEmbedder(cond_dim)

        self.noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=num_train_timesteps,
                                                       beta_schedule=beta_schedule,
                                                       prediction_type=prediction_type)


    def forward(self, batch, timesteps):
        device = batch['state_surface'].device
        bs = batch['state_surface'].shape[0]

        input_surface = torch.cat([#batch['state_surface'], 
                                   batch['state_constant'], 
                                   batch['pred_state_surface'],
                                   batch['surface_noisy']], dim=1)
        input_surface = input_surface.squeeze(-3)
        input_level = torch.cat([#batch['state_level'],
                                   batch['pred_state_level'],
                                   batch['level_noisy']], dim=1)
        
        month = torch.tensor([int(x[5:7]) for x in batch['time']]).to(device)
        month_emb = self.month_embedder(month)
        hour = torch.tensor([int(x[-5:-3]) for x in batch['time']]).to(device)
        hour_emb = self.hour_embedder(hour)

        timestep_emb = self.timestep_embedder(timesteps)

        cond_emb = month_emb + hour_emb + timestep_emb

        out = self.backbone(input_surface, input_level, cond_emb)
        
        return out
    
    def training_step(self, batch, batch_nb):
        # sample timesteps
        device = batch['state_surface'].device
        bs = batch['state_surface'].shape[0]

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

        surface_noise = torch.randn_like(batch['next_state_surface'])
        batch['surface_noisy'] = self.noise_scheduler.add_noise(batch['next_state_surface'], surface_noise, timesteps)

        level_noise = torch.randn_like(batch['next_state_level'])
        batch['level_noisy'] = self.noise_scheduler.add_noise(batch['next_state_level'], level_noise, timesteps)

        pred = self.forward(batch, timesteps)

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target_surface = surface_noise
            target_level = level_noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target_surface = self.noise_scheduler.get_velocity(batch['surface_noisy'], surface_noise, timesteps)
            target_level = self.noise_scheduler.get_velocity(batch['level_noisy'], level_noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        

        batch['next_state_surface'] = target_surface
        batch['next_state_level'] = target_level

        _, _, per_sample_loss = self.loss(pred, batch)

        # reweight loss with min snr gamma
        if self.snr_gamma:
            snr = compute_snr(self.noise_scheduler, timesteps)
            base_weight = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )

            if self.noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective needs to be floored to an SNR weight of one.
                mse_loss_weights = base_weight + 1
            else:
                # Epsilon and sample both use the same loss weights.
                mse_loss_weights = base_weight

            loss = (per_sample_loss * mse_loss_weights)
            
        loss = loss.mean()
        self.mylog(loss=loss)
        return loss
    
    def loss(self, pred, batch, lat_coeffs=lat_coeffs_equi):
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
    
        nvar = (surface_coeffs.sum().item() + 5)
        
        loss = (mse_surface_w.sum(1).mean((-3, -2, -1)) + 
                mse_level_w.sum(1).mean((-3, -2, -1)))/nvar
        
        return mse_surface, mse_level, loss
    
    def validation_step(self, batch, batch_nb):
        # for now it just logs training loss
        return self.training_step(batch, batch_nb)
        
