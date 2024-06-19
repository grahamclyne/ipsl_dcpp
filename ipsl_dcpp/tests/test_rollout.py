from ipsl_dcpp.model.ipsl_dataset import IPSL_DCPP
import torch
import lightning as pl
from ipsl_dcpp.model.pangu import PanguWeather
from hydra import compose, initialize
from omegaconf import OmegaConf

import hydra
import os
# os.environ["CUDA_VISIBLE_DEVICES"]=""
#torch.set_default_tensor_type(torch.FloatTensor)

with initialize(version_base=None, config_path="../conf"):
    cfg = compose(config_name="config")
pl.seed_everything(cfg.experiment.seed)
train = hydra.utils.instantiate(
    cfg.experiment.train_dataset,
    generate_statistics=False,
    surface_variables=cfg.experiment.surface_variables,
    depth_variables=cfg.experiment.depth_variables,
    plev_variables=cfg.experiment.plev_variables,
    work_path=cfg.environment.work_path,
    scratch_path=cfg.environment.scratch_path,
)

train_dataloader = torch.utils.data.DataLoader(
    train,
    batch_size=2,
    shuffle=True,
    num_workers=1
)

val = hydra.utils.instantiate(
    cfg.experiment.val_dataset,
    generate_statistics=False,
    surface_variables=cfg.experiment.surface_variables,
    depth_variables=cfg.experiment.depth_variables,
    plev_variables=cfg.experiment.plev_variables,
    work_path=cfg.environment.work_path,
    scratch_path=cfg.environment.scratch_path,
)

val_dataloader = torch.utils.data.DataLoader(
    val,
    batch_size=1,
    shuffle=False,
    num_workers=9
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
    max_epochs=cfg.experiment.max_epochs,
    enable_checkpointing=True,
    log_every_n_steps=1,
   # max_steps=cfg.experiment.max_steps if not cfg.debug else 10,
    precision="16-mixed",
    #precision='32',
    profiler='simple' if cfg.debug else None,
   # devices=cfg.experiment.num_gpus,
   # strategy='ddp_find_unused_parameters_true',
    #limit_train_batches=0.01 if cfg.debug else 1
    #limit_val_batches=0.01 if cfg.debug else 1,
    num_sanity_val_steps=2,
  #  device='cpu',
  #accelerator='mps',
  #CONV3D not supported by mps, have to use cpu when local 
    accelerator= 'mps' if cfg.environment.name == 'local' else 'gpu',
     limit_test_batches=0.00023540489642184556, #this is exactly one batch according to lightning
    
)
trainer.test(model, val_dataloader)