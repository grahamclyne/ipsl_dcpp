from lightning.pytorch.loggers import WandbLogger
from ipsl_dataset import IPSL_DCPP
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
import os 
import lightning.pytorch as pl
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pangu import PanguWeather


def main():
    scratch = os.environ['SCRATCH']
    wandb_logger = WandbLogger(project="pangu_ipsl_autoregression")
    #logger = TensorBoardLogger("tb_logs", name="my_model")

    train = IPSL_DCPP('train')
    val = IPSL_DCPP('val')
    train_dataloader = torch.utils.data.DataLoader(train,batch_size=32,shuffle=True,num_workers=2)
    val_dataloader = torch.utils.data.DataLoader(val,batch_size=1,shuffle=False,num_workers=2)

    #regressor = UNet2(n_channels=len(input_variables),n_out_channels=len(target_variables))
    model = PanguWeather()
    
    checkpoint_callback = ModelCheckpoint(
        filename=wandb_logger.experiment.name+"_{epoch:02d}",
        every_n_epochs=2,
        dirpath=f'{scratch}/checkpoints/',
        save_top_k=-1,
    )

  #  trainer = pl.Trainer(max_epochs=1,callbacks=[checkpoint_callback,DeviceStatsMonitor()],default_root_dir=f"{scratch}/checkpoint_ipsl_unet/",enable_checkpointing=True,log_every_n_steps=10,min_epochs=5,logger=wandb_logger)
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback],default_root_dir=f"{scratch}/checkpoint_{wandb_logger.experiment.name}",
        enable_checkpointing=True,
        log_every_n_steps=1,
        min_epochs=5,
        logger=wandb_logger,
        precision="16-mixed",
        devices=1,
        #strategy='ddp_find_unused_parameters_true',
        accelerator="gpu"
    )
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )



if __name__ == "__main__":
    
    # run on cluster
    gpus = getattr(cfg, 'gpus', 1)
    #cpus = getattr(cfg, 'cpus', 1)
    Path('sblogs').mkdir(exist_ok=True)
    exp_id = len(list(Path('sblogs').glob(f'{cfg.name}_*')))
    Path(f'sblogs/{cfg.name}_{exp_id}').mkdir()
    aex = submitit.AutoExecutor(folder=f'sblogs/{cfg.name}_{exp_id}')

    if args.jza: # run on jean zay a100
         aex.update_parameters(tasks_per_node=gpus, 
                                 nodes=1+(gpus-1)//8, 
                                 gpus_per_node=min(gpus, 8), 
                                 timeout_min=20*60,
                                 cpus_per_task=8,
                                 slurm_partition="gpu_p5",
                                 slurm_account="ehx@a100",
                                 slurm_job_name=cfg.name,
                                 slurm_constraint="a100"
                             )
    elif args.jzv:
        #cfg.batch_size = cfg.batch_size // 3
        aex.update_parameters(tasks_per_node=gpus, 
                                 nodes=1+(gpus-1)//8, 
                                 gpus_per_node=min(gpus, 8), 
                                 slurm_time="20:00:00",
                                 cpus_per_task=3,
                                 slurm_account="ehx@v100",
                                 slurm_job_name=cfg.name,
                                 slurm_constraint="v100-32g",
                                 stderr_to_stdout=True,
                                 slurm_exclusive=True,
                                 #slurm_use_srun=True,
                                 #slurm_'nomultithread'
                             )

    else: # run on cleps a100
        aex.update_parameters(tasks_per_node=args.gpus, 
                                 nodes=1, 
                                 gpus_per_node=gpus, 
                                 timeout_min=2*24*60,
                                 mem_gb=30*gpus, 
                                 cpus_per_task=8,
                                 slurm_partition='gpu',
                              slurm_constraint='a100')

    aex.submit(main, cfg, hparams)
