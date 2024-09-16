from omegaconf import DictConfig,OmegaConf
import torch
import lightning as pl
import hydra
import os
import glob

@hydra.main(config_path='./conf',config_name="config.yaml")
def main(cfg: DictConfig):
    device = 'cuda'
    pl.seed_everything(cfg.seed)
    test = hydra.utils.instantiate(
        cfg.dataloader.dataset,domain='test',debug=True
    )
    test_loader = torch.utils.data.DataLoader(test, 
                                              batch_size=1,
                                              num_workers=cfg.cluster.cpus,
                                              shuffle=False) 
    trainer = pl.Trainer(
                    limit_test_batches=1,
                    limit_predict_batches=1
                    )
    pl_module = hydra.utils.instantiate(
        cfg.module.module,
        backbone=hydra.utils.instantiate(cfg.module.backbone),
        dataset=test_loader.dataset,
        num_inference_steps=25,
        num_rollout_steps=10,
        num_ensemble_members=1,
        num_batch_examples=1
    ).to(device)
    list_of_files = glob.glob(f'{cfg.exp_dir}/checkpoints/*') 
    path = max(list_of_files)
    print(path)
    checkpoint_path = torch.load(path,map_location=torch.device('cuda'))
    pl_module.load_state_dict(checkpoint_path['state_dict'])
    torch.save(pl_module.sharpness_test(),f'{cfg.name}_sharpnesses.pt')

if __name__ == "__main__":
    main()