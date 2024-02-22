import os 
from ipsl_dataset import IPSL_DCPP
import lightning.pytorch as pl
import torch
import hydra
import numpy as np
from ipsl_dataset import surface_variables
from hydra import compose, initialize
from omegaconf import DictConfig,OmegaConf

@hydra.main(version_base=None,config_path='./conf',config_name="config.yaml")
def main(cfg: DictConfig):
    work = os.environ['WORK']
    with_soil_checkpoint_5_year = f'{work}/ipsl_dcpp/ipsl_dcpp_emulation/jvlh5yoa/checkpoints/24_month_epoch=07.ckpt'

    checkpoint_with_soil = torch.load(with_soil_checkpoint_5_year,map_location=torch.device('cpu'))
    test = IPSL_DCPP('test',cfg.experiment.lead_time_months)
    test_dataloader = torch.utils.data.DataLoader(test,batch_size=1,shuffle=False,num_workers=1)

    soil_model = hydra.utils.instantiate(cfg.experiment.module,backbone=hydra.utils.instantiate(cfg.experiment.backbone,soil=True),dataset=test_dataloader.dataset)
    soil_model.load_state_dict(checkpoint_with_soil['state_dict'])

    #do rollout
    batch = next(iter(test_dataloader))
    surfaces = []
    for i in range(10):
        print(i)
        with torch.no_grad():
            output = soil_model.forward(batch)
        batch=dict(state_surface=output['next_state_surface'],
                   state_level=output['next_state_level'],
                   state_depth=output['next_state_depth'],
                   state_constant=batch['state_constant'])
        surfaces.append(output['next_state_surface'])
    np.save('rollout.npy',np.stack(surfaces))
    
    
if __name__ == "__main__":
    main()
