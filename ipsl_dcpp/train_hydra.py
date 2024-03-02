from omegaconf import DictConfig,OmegaConf
import hydra
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint,TQDMProgressBar,ModelSummary
import submitit
import os
from pathlib import Path




def train(cfg):
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(cfg.experiment.seed)
    bar = TQDMProgressBar(refresh_rate=100)

    if(cfg.environment.name == 'jean_zay'):
        os.environ['WANDB_MODE'] = 'offline'
        os.environ['WANDB_API_KEY'] = 'c1f678c655920120ec68e1dc542a9f5bab02dbfa'
        os.environ['WANDB_DIR'] = cfg.environment.scratch_path + '/wandb'
        os.environ['WANDB_CACHE_DIR'] = cfg.environment.scratch_path + '/wandb'
        os.environ['WANDB_DATA_DIR'] = cfg.environment.scratch_path + '/wandb'
        
    wandb_logger = WandbLogger(
        project=cfg.project_name,
        name=cfg.experiment.name,
        log_model=True
    )
    wandb_logger.config = OmegaConf.to_container(
         cfg.experiment, resolve=True, throw_on_missing=True
    )
    train = hydra.utils.instantiate(
        cfg.experiment.train_dataset,
        surface_variables=cfg.experiment.surface_variables,
        depth_variables=cfg.experiment.depth_variables
    )
    val = hydra.utils.instantiate(
        cfg.experiment.val_dataset,
        surface_variables=cfg.experiment.surface_variables,
        depth_variables=cfg.experiment.depth_variables
    )
    train_dataloader = torch.utils.data.DataLoader(
        train,
        batch_size=cfg.experiment.train_batch_size,
        shuffle=True,
        num_workers=cfg.experiment.num_cpus_per_task
    )
    val_dataloader = torch.utils.data.DataLoader(
        val,
        batch_size=cfg.experiment.val_batch_size,
        shuffle=False,
        num_workers=cfg.experiment.num_cpus_per_task
    )


    checkpoint_callback = ModelCheckpoint(
        filename="{epoch:02d}",
        every_n_epochs=1,
        save_top_k=-1,
        dirpath=f"{cfg.environment.scratch_path}/checkpoint_{wandb_logger.experiment.id}/"
    )

    trainer = pl.Trainer(
        max_epochs=cfg.experiment.max_epochs,
        callbacks=[bar,checkpoint_callback,ModelSummary(max_depth=-1)],
        enable_checkpointing=True,
        log_every_n_steps=10,
       # max_steps=cfg.experiment.max_steps if not cfg.debug else 10,
        logger=wandb_logger,
        precision="16-mixed",
        profiler='simple' if cfg.debug else None,
        devices=cfg.experiment.num_gpus,
        strategy='ddp_find_unused_parameters_true' if ((cfg.experiment.num_gpus > 1) and not cfg.experiment.backbone.soil) else 'ddp' if cfg.experiment.num_gpus > 1 else 'auto',
       # strategy='ddp_find_unused_parameters_true',
        accelerator="gpu",
        #limit_train_batches=0.01 if cfg.debug else 1
        #limit_val_batches=0.01 if cfg.debug else 1,
    )

    model = hydra.utils.instantiate(
        cfg.experiment.module,
        backbone=hydra.utils.instantiate(
            cfg.experiment.backbone,
        ),
        dataset=val_dataloader.dataset
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

@hydra.main(version_base=None,config_path='./conf',config_name="config.yaml")
def main(cfg: DictConfig):
    scratch_dir = os.environ['SCRATCH']
    work_dir = os.environ['WORK']
   
    if(cfg.debug):
        cfg.experiment.train_batch_size = 1
        cfg.experiment.val_batch_size = 1
        train(cfg)
        return
    
    log_path = f'{work_dir}/submitit_logs'
    Path(log_path).mkdir(exist_ok=True)
    exp_id = len(list(Path(log_path).glob(f'{cfg.experiment.name}_*')))
    Path(f'{log_path}/{cfg.experiment.name}_{exp_id}').mkdir()
    aex = submitit.AutoExecutor(folder=f'{log_path}/{cfg.experiment.name}_{exp_id}')

    if cfg.environment.name == 'jean_zay' and cfg.experiment.gpu_type == 'a100': # run on jean zay a100
         aex.update_parameters(tasks_per_node=cfg.experiment.num_gpus, 
                                 nodes=1, 
                                 gpus_per_node=cfg.experiment.num_gpus, 
                                 #timeout_min=20*60,
                                 #timeout_min=5,
                                 slurm_time=cfg.experiment.slurm_time,
                                 cpus_per_task=cfg.experiment.num_cpus_per_task,
                                 #slurm_partition="gpu_p5",
                                 slurm_account="mlr@a100",
                                 slurm_job_name=cfg.experiment.name,
                                 slurm_constraint="a100"
                             )
    elif cfg.environment.name == 'jean_zay' and cfg.experiment.gpu_type == 'v100': # run on jean zay a100
        aex.update_parameters(tasks_per_node=cfg.experiment.num_gpus, 
                                 nodes=1, 
                                 gpus_per_node=cfg.experiment.num_gpus, 
                                 slurm_time=cfg.experiment.slurm_time,
                                 cpus_per_task=cfg.experiment.num_cpus_per_task,
                                 slurm_account="mlr@v100",
                                 slurm_job_name=cfg.experiment.name,
                                 #slurm_hint='nomultithread',
                                 
                                 slurm_constraint='v100-32g',
                                #slurm_use_srun=True,
                                # slurm_'nomultithread'
                             )

    aex.submit(train, cfg)


if __name__ == "__main__":
    main()
