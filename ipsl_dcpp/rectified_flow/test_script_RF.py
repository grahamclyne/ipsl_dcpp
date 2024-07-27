#rectified flow from https://github.com/grahamclyne/RectifiedFlow/tree/main
#load score based model AND datasets
#generate 
import torch
import lightning as pl
from hydra import compose, initialize
from omegaconf import OmegaConf
import numpy as np
import hydra
import os
import pickle
import io
import numpy as np
from matplotlib import animation
import xarray as xr 



with initialize(version_base=None, config_path="../conf"):
    cfg = compose(config_name="config")
device = 'cuda'
pl.seed_everything(cfg.seed)

train = hydra.utils.instantiate(
    cfg.dataloader.dataset,domain='train',debug=True
)
train_loader = torch.utils.data.DataLoader(train, 
                                          batch_size=1,
                                          num_workers=cfg.cluster.cpus,
                                          shuffle=False)
# train_loader.dataset.to(device)
path = f'{cfg.exp_dir}/checkpoints/checkpoint_global_step=270000.ckpt'


checkpoint_path = torch.load(path,map_location=torch.device('cuda'))
model = hydra.utils.instantiate(
    cfg.module.module,
    backbone=hydra.utils.instantiate(cfg.module.backbone),
    dataset=train_loader.dataset
)

model.load_state_dict(checkpoint_path['state_dict'])




import ipsl_dcpp.rectified_flow.sde_lib as sde_lib
sde = sde_lib.RectifiedFlow(init_type='gaussian', noise_scale=1, use_ode_sampler='rk45')
sampling_eps = 1e-3

import ipsl_dcpp.rectified_flow.losses as losses
import ml_collections
config = ml_collections.ConfigDict()

config.optim = optim = ml_collections.ConfigDict()
optim.weight_decay = 0.01
optim.optimizer = 'Adam'
optim.lr = 3e-4
optim.beta1 = 0.9
optim.eps = 1e-8
optim.warmup = 500
optim.grad_clip = 1.

optimize_fn = losses.optimization_manager(config)

train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                    reduce_mean=True, continuous=False,
                                    likelihood_weighting=False)


from ipsl_dcpp.rectified_flow.ema import ExponentialMovingAverage
#for step in range(initial_step, num_train_steps + 1):
ema = ExponentialMovingAverage(model.parameters(), decay=0.999999)
optimizer = losses.get_optimizer(config, model.parameters())
state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

for batch in train_loader:
  # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
  # batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
  # batch = batch.permute(0, 3, 1, 2)
  # batch = scaler(batch)
  # Execute one training step
    loss = train_step_fn(state, batch)
    print('loss',loss)
    