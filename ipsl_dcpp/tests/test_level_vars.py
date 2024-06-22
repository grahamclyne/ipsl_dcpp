from ipsl_dcpp.model.ipsl_dataset import IPSL_DCPP
import torch
import lightning as pl
from ipsl_dcpp.model.pangu import PanguWeather
from hydra import compose, initialize
from omegaconf import OmegaConf

import hydra
import os
os.environ['SLURM_NTASKS_PER_NODE'] = '1'

with initialize(config_path="../conf"):
    cfg = compose(config_name="config")
pl.seed_everything(cfg.experiment.seed)
train = hydra.utils.instantiate(
    cfg.experiment.train_dataset,
    generate_statistics=False,
    surface_variables=cfg.experiment.surface_variables,
    depth_variables=cfg.experiment.depth_variables,
    plev_variables=cfg.experiment.plev_variables,
    normalization='climatology',
    delta=True,
    work_path=cfg.environment.work_path,
    scratch_path=cfg.environment.scratch_path,
)

train_dataloader = torch.utils.data.DataLoader(
    train,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

#batch = next(iter(train_dataloader))
model = hydra.utils.instantiate(
    cfg.experiment.module,
    backbone=hydra.utils.instantiate(
        cfg.experiment.backbone,
    ),
    dataset=train_dataloader.dataset
)
trainer = pl.Trainer(
    enable_checkpointing=True,
    log_every_n_steps=1,
    max_steps=cfg.experiment.max_steps if not cfg.debug else 10,
    precision="16-mixed",
    #precision='32',
    profiler='simple' if cfg.debug else None,
    num_sanity_val_steps=0,
  #CONV3D not supported by mps, have to use cpu when local 
    accelerator= 'cpu' if cfg.environment.name == 'local' else 'gpu',
)

trainer.fit(
    model=model,
    train_dataloaders=train_dataloader)
