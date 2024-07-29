from omegaconf import DictConfig,OmegaConf
import hydra
import torch
import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint,TQDMProgressBar
import submitit
import os
import omegaconf
from pathlib import Path
import signal
try:
    OmegaConf.register_new_resolver("eval", eval)
except:
    pass


def get_random_code():
    import string
    import random
    # generate random code that alternates letters and numbers
    l = random.choices(string.ascii_lowercase, k=3)
    n = random.choices(string.digits, k=3)
    return ''.join([f'{a}{b}' for a, b in zip(l, n)])



class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        dirpath='./',
        save_step_frequency=50000,
        prefix="checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
        self.dirpath = dirpath

    def on_train_batch_end(self, trainer: pl.Trainer, *args, **kwargs):
        """ Check if we should save a checkpoint after every train batch """
        if not hasattr(self, 'trainer'):
            self.trainer = trainer

        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            self.save()
       
    def save(self, *args, trainer=None, **kwargs):
        if trainer is None and not hasattr(self, 'trainer'):
            print('No trainer !')
            return
        if trainer is None:
            trainer = self.trainer

        global_step = trainer.global_step
        if self.use_modelcheckpoint_filename:
            filename = trainer.checkpoint_callback.filename
        else:
            filename = f"{self.prefix}_{global_step=}.ckpt"
        ckpt_path = Path(self.dirpath) / 'checkpoints'
        ckpt_path.mkdir(exist_ok=True, parents=True)
        trainer.save_checkpoint(ckpt_path / filename)


# def train(cfg,run_id):
#     #this needs to be called on both the start up job and the running job if i wanna use variables in both settings....?
#     if(cfg.environment.name == 'jean_zay'):
#         os.environ['WANDB_MODE'] = 'offline'
#         os.environ['WANDB_API_KEY'] = 'c1f678c655920120ec68e1dc542a9f5bab02dbfa'
#         os.environ['WANDB_DIR'] = cfg.environment.scratch_path + '/wandb'
#         os.environ['WANDB_CACHE_DIR'] = cfg.environment.scratch_path + '/wandb'
#         os.environ['WANDB_DATA_DIR'] = cfg.environment.scratch_path + '/wandb'
   
#     wandb_logger = WandbLogger(
#         project=cfg.project_name,
#         name=cfg.experiment.name,
#         log_model=True,
#         id=run_id
#     )

#     job_env = submitit.JobEnvironment()
    
#     conf = OmegaConf.to_container(
#          cfg.experiment, resolve=True, throw_on_missing=True
#     )
#     if(job_env.global_rank == 0):
#         for key,value in conf.items():
#             wandb_logger.experiment.config[key] = value
#         wandb_logger.experiment.name = run_id
#    # torch.set_float32_matmul_precision('medium')
#     pl.seed_everything(cfg.experiment.seed)
#     bar = TQDMProgressBar(refresh_rate=1)
#     # train = hydra.utils.instantiate(
#     #     cfg.experiment.train_dataset,
#     #     surface_variables=cfg.experiment.surface_variables,
#     #     depth_variables=cfg.experiment.depth_variables,
#     #     plev_variables=cfg.experiment.plev_variables,
#     #     work_path=cfg.environment.work_path,
#     #     scratch_path=cfg.environment.scratch_path,
#     # )
#     # val = hydra.utils.instantiate(
#     #     cfg.experiment.val_dataset,
#     #     surface_variables=cfg.experiment.surface_variables,
#     #     depth_variables=cfg.experiment.depth_variables,
#     #     plev_variables=cfg.experiment.plev_variables,
#     #     work_path=cfg.environment.work_path,
#     #     scratch_path=cfg.environment.scratch_path,
#     # )
#     # train_dataloader = torch.utils.data.DataLoader(
#     #     train,
#     #     batch_size=cfg.experiment.train_batch_size,
#     #     shuffle=True,
#     #     num_workers=cfg.experiment.num_cpus_per_task // 2
#     # )
#     # val_dataloader = torch.utils.data.DataLoader(
#     #     val,
#     #     batch_size=cfg.experiment.val_batch_size,
#     #     shuffle=False,
#     #     num_workers=cfg.experiment.num_cpus_per_task
#     # )


#     # checkpoint_callback = ModelCheckpoint(
#     #     filename="{epoch:02d}",
#     #     every_n_epochs=1,
#     #     save_top_k=-1,
#     #     dirpath=f"{cfg.environment.scratch_path}/checkpoint_{run_id}/"
#     # )

#     trainer = pl.Trainer(
#         max_epochs=cfg.experiment.max_epochs,
#       #  callbacks=[bar,checkpoint_callback,ModelSummary(max_depth=-1)],
#         callbacks=[bar,checkpoint_callback],
#         enable_checkpointing=True,
#         log_every_n_steps=100,
#         max_steps=cfg.experiment.max_steps if not cfg.debug else 10,
#         logger=wandb_logger,
#         precision="16-mixed",
#         profiler='simple' if cfg.debug else None,
#         devices=cfg.experiment.num_gpus,
#        # strategy='ddp_find_unused_parameters_true' if ((cfg.experiment.num_gpus > 1) and not cfg.experiment.backbone.soil) else 'ddp' if cfg.experiment.num_gpus > 1 else 'auto',
#         strategy='ddp',
#         accelerator="gpu",
#         #limit_train_batches=0.01 if cfg.debug else 1
#         limit_val_batches=0.00
#     )

#     # model = hydra.utils.instantiate(
#     #     cfg.experiment.module,
#     #     backbone=hydra.utils.instantiate(
#     #         cfg.experiment.backbone,
#     #     ),
#     #     dataset=val_dataloader.dataset
#     # )

#     trainer.fit(
#         model=model,
#         train_dataloaders=train_dataloader,
#         val_dataloaders=val_dataloader
#     )

#@hydra.main(version_base=None,config_path='./conf',config_name="config.yaml")
@hydra.main(config_path='./conf',config_name="config.yaml")
def main(cfg: DictConfig):
    
    import uuid
    run_id = str(uuid.uuid4()).split('-')[0]

#    scratch_dir = os.environ['SCRATCH']
    work_dir = os.environ['WORK']


    main_node = int(os.environ.get('SLURM_PROCID', 0)) == 0
    print('is main node', main_node)
    logger = None
    ckpt_path = None

    # delete submitit handler to let PL take care of resuming
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


    # first, check if exp exists
    if Path(cfg.exp_dir).exists():
        print('Experiment already exists. Trying to resume it.')
        # exp_cfg = OmegaConf.load(Path(cfg.exp_dir) / 'config.yaml')
        # if cfg.resume:
        #     cfg = exp_cfg
        # else:
        #     # check that new config and old config match
        #     if OmegaConf.to_yaml(cfg.module, resolve=True) != OmegaConf.to_yaml(exp_cfg.module):
        #         print('Module config mismatch. Exiting')
        #         print('Old config', OmegaConf.to_yaml(exp_cfg.module))
        #         print('New config', OmegaConf.to_yaml(cfg.module))
                
        #     if OmegaConf.to_yaml(cfg.dataloader, resolve=True) != OmegaConf.to_yaml(exp_cfg.dataloader):
        #         print('Dataloader config mismatch. Exiting.')
        #         print('Old config', OmegaConf.to_yaml(exp_cfg.dataloader))
        #         print('New config', OmegaConf.to_yaml(cfg.dataloader))

            
        # trying to find checkpoints
        ckpt_dir = Path(cfg.exp_dir).joinpath('checkpoints')
        if ckpt_dir.exists():
            ckpts = list(sorted(ckpt_dir.iterdir(), key=os.path.getmtime))
            if len(ckpts):
                print('Found checkpoints', ckpts)
                ckpt_path = ckpts[-1]  


    if cfg.log:
        os.environ['WANDB_DISABLE_SERVICE'] = 'True'
        print('wandb mode', cfg.cluster.wandb_mode)
        print('wandb service', os.environ.get('WANDB_DISABLE_SERVICE', 'variable unset'))
        run_id = cfg.name + '-'+get_random_code() if cfg.cluster.manual_requeue else cfg.name
        logger = pl.loggers.WandbLogger(project=cfg.project,
                                        name=cfg.name,
                                        id=run_id,
                                        save_dir=cfg.cluster.wandb_dir,
                                        offline=(cfg.cluster.wandb_mode != 'online'))
        wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    if cfg.log and main_node and not Path(cfg.exp_dir).exists():
        print('registering exp on main node')
        hparams = OmegaConf.to_container(cfg, resolve=True)
        print(hparams)
        wandb.config = hparams
        logger.log_hyperparams(hparams)
        Path(cfg.exp_dir).mkdir(parents=True)
        with open(Path(cfg.exp_dir) / 'config.yaml', 'w') as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))

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

    pl_module = hydra.utils.instantiate(
        cfg.module.module,
        backbone=hydra.utils.instantiate(cfg.module.backbone),
        dataset=train_loader.dataset
    )
    if hasattr(cfg, 'load_ckpt'):
        # load weights w/o resuming run
        pl_module.init_from_ckpt(cfg.load_ckpt)


    checkpointer = CheckpointEveryNSteps(dirpath=cfg.exp_dir,
                                         save_step_frequency=cfg.save_step_frequency)

    print('Manual submitit Requeuing')

    def handler(*args, **kwargs):
        print('GCO: SIGTERM signal received. Requeueing job on main node.')
        if main_node:
            checkpointer.save()
            from submit import main as submit_main
            if cfg.cluster.manual_requeue:
                submit_main(cfg)
        exit()
        
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, handler)




    # trainer = pl.Trainer(
    #     max_epochs=cfg.experiment.max_epochs,
    #   #  callbacks=[bar,checkpoint_callback,ModelSummary(max_depth=-1)],
    #     callbacks=[bar,checkpoint_callback],
    #     enable_checkpointing=True,
    #     log_every_n_steps=100,
    #     max_steps=cfg.experiment.max_steps if not cfg.debug else 10,
    #     logger=wandb_logger,
    #     precision="16-mixed",
    #     profiler='simple' if cfg.debug else None,
    #     devices=cfg.experiment.num_gpus,
    #    # strategy='ddp_find_unused_parameters_true' if ((cfg.experiment.num_gpus > 1) and not cfg.experiment.backbone.soil) else 'ddp' if cfg.experiment.num_gpus > 1 else 'auto',
    #     strategy='ddp',
    #     accelerator="gpu",
    #     #limit_train_batches=0.01 if cfg.debug else 1
    #     limit_val_batches=0.00
    # )


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
                callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=100),
                           checkpointer], 
                gradient_clip_val=1,

                accumulate_grad_batches=cfg.accumulate_grad_batches,
                logger=logger,
                plugins=[],
                limit_val_batches=0 if cfg.debug else 1,
              #  limit_val_batches=cfg.limit_val_batches, # max 5 samples
                limit_train_batches=1 if cfg.debug else None
                )
    
    if cfg.debug:
        breakpoint()

    trainer.fit(pl_module, train_loader,val_loader,ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
