#evaluation 
#take in checkpoint and sample ~1000 batches and give stats
#5 member rollout on several batches - stats on the rollouts, with gifs of several rollouts
#this could 
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



with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config")
device = 'cuda'
pl.seed_everything(cfg.seed)

test = hydra.utils.instantiate(
    cfg.dataloader.dataset,domain='test',debug=True
)
test_loader = torch.utils.data.DataLoader(test, 
                                          batch_size=1,
                                          num_workers=cfg.cluster.cpus,
                                          shuffle=False) 
pl_module = hydra.utils.instantiate(
    cfg.module.module,
    backbone=hydra.utils.instantiate(cfg.module.backbone),
    dataset=test_loader.dataset
).to(device)

path = f'{cfg.exp_dir}/checkpoints/checkpoint_global_step=37189.ckpt'
#path = '/gpfsscratch/rech/mlr/udy16au/model_output/ipsl_diffusion/flow-r6u1g0/checkpoints/epoch=12-step=30000.ckpt'
# checkpoint_path = torch.load(path,map_location=torch.device('cuda'))
# pl_module.load_state_dict(checkpoint_path['state_dict'])

trainer = pl.Trainer(
                limit_test_batches=1,
                limit_predict_batches=1
                )
#trainer.test(pl_module,test_loader,ckpt_path=path)

# test_loader = torch.utils.data.DataLoader(test, 
#                                           batch_size=5,
#                                           num_workers=cfg.cluster.cpus,
#                                           shuffle=False) 
trainer.predict(pl_module,test_loader,ckpt_path=path)
