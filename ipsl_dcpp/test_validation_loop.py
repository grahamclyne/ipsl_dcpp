from omegaconf import DictConfig,OmegaConf
import hydra
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint,TQDMProgressBar
import submitit
import os
from pathlib import Path
import signal
try:
    OmegaConf.register_new_resolver("eval", eval)
except:
    pass


#@hydra.main(version_base=None,config_path='./conf',config_name="config.yaml")
@hydra.main(config_path='./conf',config_name="config.yaml")
def main(cfg: DictConfig):
    
    train = hydra.utils.instantiate(
        cfg.dataloader.dataset,domain='train'
    )
    val = hydra.utils.instantiate(
        cfg.dataloader.dataset,domain='val'
    )
    val_loader = torch.utils.data.DataLoader(val, 
                                              batch_size=cfg.batch_size,
                                              num_workers=cfg.cluster.cpus,
                                              shuffle=False) 
    train_loader = torch.utils.data.DataLoader(train, 
                                              batch_size=cfg.batch_size,
                                              num_workers=cfg.cluster.cpus,
                                              shuffle=True) 
    os.environ['WANDB_DISABLE_SERVICE'] = 'True'
    run_id = cfg.name +'test' if cfg.cluster.manual_requeue else cfg.name
    logger = pl.loggers.WandbLogger(project=cfg.project,
                                    name=cfg.name,
                                    id=run_id,
                                    save_dir=cfg.cluster.wandb_dir,
                                    offline=(cfg.cluster.wandb_mode != 'online'))
        

    pl_module = hydra.utils.instantiate(
        cfg.module.module,
        backbone=hydra.utils.instantiate(cfg.module.backbone),
        dataset=train_loader.dataset
    )
  

    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(cfg.seed)
    trainer = pl.Trainer(
                devices="auto",
                accelerator="auto",
                #strategy="auto",
                strategy="ddp_find_unused_parameters_true",
             #   strategy="ddp",
                precision=cfg.cluster.precision,
                log_every_n_steps=cfg.log_freq, 
                profiler=getattr(cfg, 'profiler', None),
                max_steps=cfg.max_steps,
                callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=100)], 
                gradient_clip_val=1,

                accumulate_grad_batches=cfg.accumulate_grad_batches,
                logger=logger,
                plugins=[],
                limit_val_batches=1,
              #  limit_val_batches=cfg.limit_val_batches, # max 5 samples
                limit_train_batches=1 
                )
    

    trainer.fit(pl_module, train_loader,val_loader)

main()