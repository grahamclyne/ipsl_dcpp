import os 
from ipsl_dataset import IPSL_DCPP
import lightning.pytorch as pl
import torch
import hydra
import numpy as np
from ipsl_dataset import surface_variables
from hydra import compose, initialize
from omegaconf import DictConfig,OmegaConf
import datetime

def inc_time(batch_time):
    batch_time = datetime.datetime.strptime(batch_time,'%Y-%m')
    if(batch_time.month == 12):
        year = batch_time.year + 1
        month = 1
    else:
        year = batch_time.year
        month = batch_time.month + 1
    return f'{year}-{month}'
    
def rollout(rollout_length,dataloader):
    iter_dl = iter(dataloader)
    surfaces = []
    plevs = []
    model_plevs = []
    model_surfaces = []
    for i in range(rollout_length):
        batch_actual = next(iter_dl)
        if(i == 0):
            batch = batch_actual
        model_surfaces.append(batch_actual['next_state_surface'])
        #model_plevs.append(batch_actual['next_state_level'])
        print(batch['time'])

       # print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
        with torch.no_grad():
            output = soil_model.forward(batch)

       # output['next_state_surface'][:,var_index] = torch.where(land_mask == 1,output['next_state_surface'][:,var_index],0)
        batch=dict(state_surface=output['next_state_surface'],
                   #state_level=output['next_state_level'] + batch['state_level'], 
                   state_depth=output['next_state_depth'],
                   state_constant=batch['state_constant'],
                   next_state_surface=output['next_state_surface'],
                   next_state_depth=output['next_state_depth'],
                  time=[inc_time(batch['time'][0])])
        surfaces.append(batch['state_surface'])
    #    plevs.append(output['next_state_level'])
    return surfaces,model_surfaces,batch,batch_actual

@hydra.main(version_base=None,config_path='./conf',config_name="config.yaml")
def main(cfg: DictConfig):
    work = os.environ['WORK']
    checkpoint = f'{work}/ipsl_dcpp/ipsl_dcpp_emulation/ht87tji5/checkpoints/24_month_epoch=01.ckpt'

    checkpoint = torch.load(checkpoint,map_location=torch.device('cpu'))
    test = IPSL_DCPP('test',cfg.experiment.lead_time_months)
    test_dataloader = torch.utils.data.DataLoader(test,batch_size=1,shuffle=False,num_workers=4)

    soil_model = hydra.utils.instantiate(cfg.experiment.module,backbone=hydra.utils.instantiate(cfg.experiment.backbone),dataset=test_dataloader.dataset)
    soil_model.load_state_dict(checkpoint['state_dict'])
    #print(soil_model)
    #do rollout
    surfaces,model_surfaces,batch,batch_actual = rollout(12,test_dataloader)
if __name__ == "__main__":
    main()
