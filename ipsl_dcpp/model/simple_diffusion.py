
import lightning.pytorch as pl
import diffusers
import torch
from ipsl_dcpp.model.pangu import TimestepEmbedder

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
        num_cycles
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.month_embedder = TimestepEmbedder(256)
        self.timestep_embedder = TimestepEmbedder(256)
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.backbone = backbone # necessary to put it on device
        self.noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=num_diffusion_timesteps,
                                                       beta_schedule='squaredcos_cap_v2',
                                                       beta_start=0.0001,
                                                       beta_end=0.012,
                                                       prediction_type='sample',
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
    
        # elif self.conditional == 'state_only':
        #     input_surface = torch.cat([batch['state_surface']*sel, 
        #                             batch['state_constant'], 
        #                             batch['surface_noisy']], dim=1)
        #     input_level = torch.cat([batch['state_level']*sel,
        #                             batch['level_noisy']], dim=1) 
        # input_surface = input_surface.squeeze(-3)
        month = torch.tensor([int(x[5:7]) for x in batch['time']]).to(device)
        month_emb = self.month_embedder(month)
        timestep_emb = self.timestep_embedder(timesteps)
        cond_emb = month_emb + timestep_emb
        out = self.backbone(batch, cond_emb)
        return out

    def training_step(self, batch, batch_nb):
        # sample timesteps
        device = batch['state_surface'].device
        bs = batch['state_surface'].shape[0]

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

        surface_noise = torch.randn_like(batch['next_state_surface'])
        level_noise = torch.randn_like(batch['next_state_level'])

        batch['surface_noisy'] = self.noise_scheduler.add_noise(batch['next_state_surface'], surface_noise, timesteps)
        #batch['level_noisy'] = self.noise_scheduler.add_noise(batch['next_state_level'], level_noise, timesteps)

        batch['surface_noisy'] = self.noise_scheduler.scale_model_input(batch['surface_noisy'])
        #batch['level_noisy'] = self.noise_scheduler.scale_model_input(batch['level_noisy'])

        # Get the target for loss
        target_surface = batch['next_state_surface']
       # target_level = batch['next_state_level']
        
        # create uncond
        sel = (torch.rand((bs,), device=device))
        pred = self.forward(batch, timesteps, sel.float()[:, None, None, None, None])
        # compute loss
        batch['next_state_surface'] = target_surface
        #batch['next_state_level'] = target_level

        _, _, per_sample_loss_cond = self.loss(pred, batch)

        loss = per_sample_loss_cond
        self.mylog(cond_loss=per_sample_loss_cond.mean())
        loss = loss.mean()
        self.mylog(loss=loss)

        return loss


    def loss(self, pred, batch, lat_coeffs=None):
            if lat_coeffs is None:
                lat_coeffs = self.dataset.lat_coeffs_equi
        # surface_coeffs = pangu_surface_coeffs
            device = batch['next_state_level'].device
            mse_surface = (pred['next_state_surface'] - batch['next_state_surface']).pow(2)

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
            loss = mse_surface_w.sum(1).mean((-3, -2, -1))
            return mse_surface, _, loss
            
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