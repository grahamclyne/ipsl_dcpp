import torch
import lightning as pl
from hydra import compose, initialize
import pickle
import hydra
import os
import numpy as np
import pandas as pd

with initialize(version_base=None, config_path="../conf"):
    cfg = compose(config_name="config")
device = 'cuda'
pl.seed_everything(cfg.seed)

val = hydra.utils.instantiate(
    cfg.dataloader.dataset,domain='val'
)
val_loader = torch.utils.data.DataLoader(val, 
                                          batch_size=1,
                                          num_workers=0,
                                          shuffle=True) #as long as we seed the same, will get the same as the batches
pl_module = hydra.utils.instantiate(
    cfg.module.module,
    backbone=hydra.utils.instantiate(cfg.module.backbone),
    dataset=val_loader.dataset
).to(device)


#x = np.stack(val.timestamps)[:,2]
path = '/gpfsscratch/rech/mlr/udy16au/wandb/ipsl_diffusion/epsilon_no_conv_head_wd_actual-u1a2l9/checkpoints/epoch=10-step=64746.ckpt.ckpt'
#path = f'{cfg.exp_dir}/checkpoints/checkpoint_global_step=60000.ckpt'
checkpoint_path = torch.load(path,map_location=torch.device('cuda'))
#checkpoint_path = torch.load(f'epoch=45.ckpt',map_location=torch.device('mps'))
pl_module.load_state_dict(checkpoint_path['state_dict'])
#inv_map = {v: k for k, v in val.id2pt.items()}
iter_val = iter(val_loader)
# if(cfg.module.backbone.plev):
#     batch['state_level'] = batch['state_level'].to(device)
#     batch['prev_state_level'] = batch['prev_state_level'].to(device)
#     batch['next_state_level'] = batch['next_state_level'].to(device)

# batch['state_surface'] = batch['state_surface'].to(device)
# batch['state_constant']= batch['state_constant'].to(device)
# batch['prev_state_surface'] =batch['prev_state_surface'].to(device)
# batch['next_state_surface'] = batch['next_state_surface'].to(device)
# batch['forcings'] = batch['forcings'].to(device)
# batch['solar_forcings'] = batch['solar_forcings'].to(device)

#batch = {batch[k].to(device) if (k != 'time') and (k != 'next_time') and (k !=  else v  for k, v in batch.items()}
lead_time_months=10
num_ensemble_members=1
for i in range(num_ensemble_members):
    #indices = np.stack(val.timestamps)[np.where((pd.to_datetime(x).year == 2001) & (pd.to_datetime(x).month == 1),True,False)][:,[0,1]]
    # batch = next(iter_val)
    print(i,'ensemble member')
   # index = inv_map[(indices[i][0],indices[i][1])]
    batch = val.__getitem__(118*i)
    batch['state_constant'] = torch.Tensor(batch['state_constant'])
    batch['time'] = [batch['time']]
    batch['next_time'] = [batch['next_time']]
    print(batch['time'])
    batch = {k: v.unsqueeze(0) if (k != 'time') and (k != 'next_time') else v for k, v in batch.items()}

    if(cfg.module.backbone.plev):
        batch['state_level'] = batch['state_level'].to(device)
        batch['prev_state_level'] = batch['prev_state_level'].to(device)
        batch['next_state_level'] = batch['next_state_level'].to(device)



    # maximum = torch.quantile(batch['state_surface'][0,:].reshape(34,-1),0.999,dim=1).to(device)
    # minimum = torch.quantile(batch['state_surface'][0,:].reshape(34,-1),0.001,dim=1).to(device)

    # maximum = maximum.unsqueeze(1).unsqueeze(2).expand(-1,143,144).unsqueeze(0)
    # minimum = minimum.unsqueeze(1).unsqueeze(2).expand(-1,143,144).unsqueeze(0)

    batch['state_surface'] = batch['state_surface'].to(device)
    batch['state_constant']= batch['state_constant'].to(device)
    batch['prev_state_surface'] = batch['prev_state_surface'].to(device)
    batch['next_state_surface'] = batch['next_state_surface'].to(device)
    batch['forcings'] = batch['forcings'].to(device)
    batch['solar_forcings'] = batch['solar_forcings'].to(device)

    output = pl_module.sample_rollout(batch, lead_time_months=lead_time_months,seed = i)
    file_name = f'{cfg.exp_dir}/rollouts/{i}_rollout_{cfg.name}.pkl'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name,'wb') as f:
        pickle.dump(output,f)
