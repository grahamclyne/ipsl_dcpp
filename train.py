from lightning.pytorch.loggers import WandbLogger
from ipsl_dataset import IPSL_DCPP
import torch
from unet import UNet2
from lightning.pytorch.callbacks import ModelCheckpoint
import os 
from ipsl_dataset import input_variables
import lightning.pytorch as pl
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.loggers import TensorBoardLogger


def main():
    scratch = os.environ['SCRATCH']
    wandb_logger = WandbLogger(project="unet_ipsl_autoregression")
    logger = TensorBoardLogger("tb_logs", name="my_model")

    train = IPSL_DCPP('train')
    val = IPSL_DCPP('val')
    train_dataloader = torch.utils.data.DataLoader(train,batch_size=4,shuffle=True,num_workers=3)
    val_dataloader = torch.utils.data.DataLoader(val,batch_size=1,shuffle=False,num_workers=3)

    regressor = UNet2(n_channels=len(input_variables))
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
        log_every_n_steps=10,
        min_epochs=5,
        logger=logger,
        precision="16-mixed",
    )
    trainer.fit(
        model=regressor, 
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )



if __name__ == "__main__":
    main()