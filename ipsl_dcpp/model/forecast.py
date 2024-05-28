import lightning.pytorch as pl
import torch
import diffusers
#from ipsl_dataset import surface_variables
lat_coeffs_equi = torch.tensor([torch.cos(x) for x in torch.arange(-torch.pi/2, torch.pi/2, torch.pi/143)])
lat_coeffs_equi =  (lat_coeffs_equi/lat_coeffs_equi.mean())[None, None, None, :, None]
#pressure_levels = torch.tensor([  50,  100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925,1000]).float()
#surface_coeffs = torch.tensor([0.1, 0.1, 1.0, 0.1])[None, :, None, None, None] # graphcast
#surface_coeffs = torch.tensor([0.25, 0.25, 0.25, 0.25])[None, :, None, None, None] # pangu coeffs
#level_coeffs = (pressure_levels/pressure_levels.mean())[None, None, :, None, None]

class ForecastModule(pl.LightningModule):
    def __init__(self, 
                 backbone,
                 dataset,
                 delta,
                 lr, 
                 betas,
                 weight_decay,
                 num_warmup_steps, 
                 num_training_steps,
                 num_cycles,
                ):

        super().__init__()
        self.__dict__.update(locals())
        self.backbone = backbone # necessary to put it on device
       # self.climatology = torch.from_numpy(np.load('../data/climatology_from_train.npy'))
        self.save_hyperparameters(ignore=['backbone'])
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
        
    def forward(self, x):
        return self.backbone(x)
            
    def mylog(self, dct={}, **kwargs):
        mode = 'train_' if self.training else 'val_'
        dct.update(kwargs)
        for k, v in dct.items():
            self.log(mode+k, v, prog_bar=True,sync_dist=True)
            
    def loss(self, pred, batch, lat_coeffs=lat_coeffs_equi):            
        device = batch['next_state_surface'].device
        mse_surface = (pred['next_state_surface'] - batch['next_state_surface']).abs().pow(2)
        mse_level = (pred['next_state_level'] - batch['next_state_level']).abs().pow(2)

        #if mse_surface.shape[-2] == 128:
        #    mse_surface = mse_surface[..., 4:-4, 8:-8]
        mse_surface = mse_surface.mul(lat_coeffs.to(device)) # latitude coeffs
        mse_level = mse_level.mul(lat_coeffs.to(device)) # latitude coeffs

        #mse_surface_w = mse_surface.mul(surface_coeffs.to(device))
     #   mse_level = (pred['next_state_level'] - batch['next_state_level']).pow(2)
       # if mse_level.shape[-2] == 128:
       #     mse_level = mse_level[..., 4:-4, 8:-8]
      #  mse_level = mse_level.mul(lat_coeffs.to(device))
       # mse_level_w = mse_level.mul(level_coeffs.to(device))
        if(self.backbone.soil):
       # nvar = (surface_coeffs.sum().item() + 5)
            mse_depth = (pred['next_state_depth'] - batch['next_state_depth']).pow(2)
            mse_depth = mse_depth.mul(lat_coeffs.to(device))
        #loss = (mse_surface.sum(1).mean() + mse_level.sum(1).mean())/nvar
#            loss = (mse_surface.sum(1).mean() + mse_level.sum(1).mean() + mse_depth.sum(1).mean())
            loss = (mse_surface.sum(1).mean() + mse_depth.sum(1).mean() + mse_level.sum(1).mean())
            return mse_surface,mse_depth,loss

        else:
           # loss = (mse_surface.sum(1).mean() + mse_level.sum(1).mean())
            loss = mse_surface.sum(1).mean() 
            return mse_surface,None,loss

    def training_step(self, batch, batch_nb):
        pred = self.forward(batch)
       # if self.delta:
       #     pred = dict(#next_state_level=batch['state_level']+pred['next_state_level'],
       #                next_state_surface=batch['state_surface']+pred['next_state_surface'],
       #                next_state_depth=batch['state_depth']+pred['next_state_depth'])
        _, _, loss = self.loss(pred, batch)
       # print('pred',pred)
       # print('loss',loss)
        self.mylog(loss=loss)
        
     #   loss.backward()
     #   for name, param in self.backbone.named_parameters():
     #       if param.grad is None:
     #           print(name)
     #   print('used')
     #   for name, param in self.backbone.named_parameters():
     #       if param.grad is not None:
     #           print(name)
        return loss
        
        
    def validation_step(self, batch, batch_nb):
        pred = self.forward(batch)
        _, _, loss = self.loss(pred, batch)
        self.mylog(loss=loss)
        # to compare with batch['next_state_level']...
     #   pred, batch = self.dataset.denormalize(pred, batch)
     #   mse_surface, mse_level, _ = self.loss(pred, batch)
     #   mse_surface,mse_depth,_ = self.loss(pred,batch)
     #   mse_level = mse_level.cpu().mean((-2, -1)).sqrt().mean(0)
     #   mse_surface = mse_surface.cpu().mean((-3, -2, -1)).sqrt().mean(0)

        #headline = (mse_level[0, 7]/100 + mse_level[3, 10] + mse_level[4, 9] 
        #        + (mse_level[1, 9] + mse_level[2, 9])/3/2**.5
        #        + mse_surface[2] + mse_surface[0] + mse_surface[3]/100)/7

    
       # self.mylog(batch_size=pred['next_state_surface'].shape[0]*1.0)
                        #  gpp=surface_variables.index('gpp'),  
                        #   npp=surface_variables.index('npp'), 
                        #   nep=surface_variables.index('nep'))
                          # headline=headline)
        return loss

    def predict_step(self,batch,batch_nb):
        pred = self.forward(batch)
        return pred,batch
    
    
    
    def configure_optimizers(self):
        print('configure optimizers')
        opt = torch.optim.AdamW(self.backbone.parameters(), 
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
