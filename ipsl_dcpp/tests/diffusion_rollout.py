from ipsl_dcpp.model.ipsl_dataset import IPSL_DCPP
import torch
import lightning as pl
from ipsl_dcpp.model.pangu import PanguWeather
from hydra import compose, initialize
from omegaconf import OmegaConf
import pickle
import hydra
import os
import numpy as np
import pandas as pd

with initialize(version_base=None, config_path="../conf"):
    cfg = compose(config_name="config")
pl.seed_everything(cfg.experiment.seed)

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
    num_workers=1
)

#batch = next(iter(train_dataloader))
model = hydra.utils.instantiate(
    cfg.experiment.module,
    backbone=hydra.utils.instantiate(
        cfg.experiment.backbone,
    ),
    dataset=val_dataloader.dataset

)

# trainer.logged_metrics
x = np.stack(val.timestamps)[:,2]
indices = np.stack(val.timestamps)[np.where((pd.to_datetime(x).year == 2001) & (pd.to_datetime(x).month == 1),True,False)][:,[0,1]]

scratch = os.environ['SCRATCH']
checkpoint_path = torch.load(f'{scratch}/checkpoint_bfb59d6d/epoch=16.ckpt',map_location=torch.device('cpu'))
model.load_state_dict(checkpoint_path['state_dict'])
# trainer.test(model, val_dataloader)
inv_map = {v: k for k, v in val.id2pt.items()}


for i in range(9):
    print(i,'ensemble member')
    index = inv_map[(indices[i][0],indices[i][1])]
    batch = val.__getitem__(index)
    batch['state_constant'] = torch.Tensor(batch['state_constant'])
    batch['time'] = [batch['time']]
    batch['next_time'] = [batch['next_time']]
    print(batch['time'])
    batch = {k: v.unsqueeze(0) if (k != 'time') and (k != 'next_time') else v for k, v in batch.items()}
    output = model.sample_rollout(batch, lead_time_months=120,seed = 0)
    with open(f'{i}_rollout_from_diff_ensemble_with_t_and_prev.pkl','wb') as f:
        pickle.dump(output,f)