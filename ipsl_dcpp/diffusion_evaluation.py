#evaluation 
#take in checkpoint and sample ~1000 batches and give stats
#5 member rollout on several batches - stats on the rollouts, with gifs of several rollouts
#this could 
import torch
import lightning as pl
from hydra import compose, initialize
import hydra
import os
import glob


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

#
list_of_files = glob.glob(f'{cfg.exp_dir}/checkpoints/*') 
#list_of_files = glob.glob(f'/gpfsscratch/rech/mlr/udy16au/model_output/ipsl_diffusion/flow_elevation_scaled_250_timesteps/checkpoints/*') 
path = max(list_of_files)
#path = '/gpfsscratch/rech/mlr/udy16au/model_output/ipsl_diffusion/flow_skip_smaller_embed-p3v4l5/checkpoints/epoch=2-step=4416.ckpt'
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
