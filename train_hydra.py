from omegaconf import DictConfig,OmegaConf
import hydra
import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

@hydra.main(version_base=None,config_path='./conf',config_name="config.yaml")
def main(cfg: DictConfig):

    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(cfg.experiment.seed)
    bar = pl.callbacks.TQDMProgressBar(refresh_rate=100)

    if(cfg.environment == 'jean_zay'):
        os.environ['WANDB_MODE'] = 'offline'

    wandb_logger = WandbLogger(project=cfg.project_name,name=cfg.experiment.name)
    wandb_logger.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    
    train = hydra.utils.instantiate(cfg.experiment.train_dataset)
    val = hydra.utils.instantiate(cfg.experiment.val_dataset)
    train_dataloader = torch.utils.data.DataLoader(train,batch_size=cfg.experiment.train_batch_size,shuffle=True,num_workers=cfg.experiment.num_cpus)
    val_dataloader = torch.utils.data.DataLoader(val,batch_size=cfg.experiment.val_batch_size,shuffle=False,num_workers=cfg.experiment.num_cpus)



 #   checkpoint_callback = ModelCheckpoint(
        # dirpath=checkpoints_path, # <--- specify this on the trainer itself for version control
        # filename="unet_classifier_" + wandb_logger.experiment.name + "_{epoch:02d}",
        #filename=cfg.experiment.name + "_{epoch:02d}",

        # filename="unet_classifier_" + "_{epoch:02d}",
    #    every_n_epochs=5,
        #dirpath=f'{cfg.environment.scratch_path}/checkpoints/',
    #    save_top_k=-1,)

    trainer = pl.Trainer(
        max_epochs=cfg.experiment.max_epochs,
        callbacks=[bar],
        default_root_dir=f"{cfg.environment.scratch_path}/checkpoint_{cfg.experiment.name}/",
        enable_checkpointing=True,
        log_every_n_steps=10, 
        max_steps=cfg.experiment.max_steps,
        logger=wandb_logger,
        precision="16-mixed",
        profiler='simple' if cfg.debug else None,

    )

    model = hydra.utils.instantiate(cfg.experiment.module,backbone=hydra.utils.instantiate(cfg.experiment.backbone),dataset=val_dataloader.dataset)

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

if __name__ == "__main__":
    main()