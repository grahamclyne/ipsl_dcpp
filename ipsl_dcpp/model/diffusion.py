import torch
import diffusers
from ipsl_dcpp.model.pangu import TimestepEmbedder


from .forecast import ForecastModule, surface_coeffs, level_coeffs, lat_coeffs_equi

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
        self.month_embedder = TimestepEmbedder(cond_dim)
        self.timestep_embedder = TimestepEmbedder(cond_dim)

        self.noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=num_train_timesteps,
                                                       beta_schedule=beta_schedule,
                                                       prediction_type=prediction_type)


    def forward(self, batch, timesteps):
        device = batch['state_surface'].device
        bs = batch['state_surface'].shape[0]

        input_surface = torch.cat([batch['state_surface'], 
                                   batch['state_constant'], 
                                   batch['pred_state_surface'],
                                   batch['surface_noisy']], dim=1)
        input_surface = input_surface.squeeze(-3)
        input_level = torch.cat([batch['state_level'],
                                   batch['pred_state_level'],
                                   batch['level_noisy']], dim=1)
        
        month = torch.tensor([int(x[5:7]) for x in batch['time']]).to(device)
        month_emb = self.month_embedder(month)


        timestep_emb = self.timestep_embedder(timesteps)

        cond_emb = month_emb + hour_emb + timestep_emb

        dt = np.vectorize(datetime.datetime.strptime)(batch['time'],'%Y-%m')
        time_step_conversion = np.vectorize(lambda x: x.month)
        timestep = torch.Tensor(time_step_conversion(dt)).to(surface.device)
        
        c = self.time_embedding(timestep)
        #pos_embs = self.positional_embeddings[None].expand((surface.shape[0], *self.positional_embeddings.shape))
        surface = self.patchembed2d(surface)
        upper_air = self.plev_patchembed3d(upper_air)
        depth = self.depth_patchembed3d(depth)
        x = torch.concat([surface.unsqueeze(2),depth,upper_air], dim=2)
        return x
    
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
        
