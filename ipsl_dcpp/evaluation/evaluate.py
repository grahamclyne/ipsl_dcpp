import os 
from ipsl_dataset import IPSL_DCPP
import lightning.pytorch as pl
import torch
import hydra
from ipsl_dataset import surface_variables
import matplotlib.pyplot as plt
from omegaconf import DictConfig,OmegaConf
def get_time_series(output,var_index):
    pred_means = []
    batch_means = []
    for i in range(len(output)):
        pred = output[i][0]
        batch = output[i][1]
        pred_means.append(pred[:,var_index].flatten())
        batch_means.append(batch[:,var_index].flatten())
    pred_means = torch.concat(pred_means)
    batch_means = torch.concat(batch_means)
    return pred_means,batch_means

@hydra.main(version_base=None,config_path='./conf',config_name="config.yaml")
def main(cfg: DictConfig):
    scratch = os.environ['SCRATCH']
    work = os.environ['WORK']
    run_name = ''
    #6 month 
    checkpoint_path = f'{work}/ipsl_dcpp/ipsl_dcpp_emulation/6l0y0nym/checkpoints/epoch=4-step=14125.ckpt'
    # 2 year
    checkpoint_path = f'{work}/ipsl_dcpp/ipsl_dcpp_emulation/0qo4adab/checkpoints/24_month_epoch=04.ckpt'

    no_soil_checkpoint_5_year = f'{work}/ipsl_dcpp/ipsl_dcpp_emulation/kwsn8y3n/checkpoints/24_month_epoch=07.ckpt'
    with_soil_checkpoint_5_year = f'{work}/ipsl_dcpp/ipsl_dcpp_emulation/jvlh5yoa/checkpoints/24_month_epoch=07.ckpt'
    
    no_soil_checkpoint = torch.load(no_soil_checkpoint_5_year,map_location=torch.device('cpu'))
    soil_checkpoint = torch.load(with_soil_checkpoint_5_year,map_location=torch.device('cpu'))
    print(cfg)
    print(cfg.experiment)
    test = IPSL_DCPP('test',cfg.experiment.lead_time_months)
    test_dataloader = torch.utils.data.DataLoader(test,batch_size=4,shuffle=False,num_workers=1)
    soil_model = hydra.utils.instantiate(
        cfg.experiment.module,
        backbone=hydra.utils.instantiate(cfg.experiment.backbone,soil=True),
        dataset=test_dataloader.dataset
    )
    no_soil_model = hydra.utils.instantiate(
        cfg.experiment.module,
        backbone=hydra.utils.instantiate(cfg.experiment.backbone,soil=False),
        dataset=test_dataloader.dataset
    )
    soil_model.load_state_dict(soil_checkpoint['state_dict'])
    no_soil_model.load_state_dict(no_soil_checkpoint['state_dict'])
    trainer = pl.Trainer()
    soil_output = trainer.predict(soil_model, test_dataloader)
    no_soil_output = trainer.predict(soil_model, test_dataloader)

    np.save('soil_test_output.npy',soil_output)
    np.save('no_soil_test_output.npy',no_soil_output)

    #var_name = 'tas'
    #ps,bs = get_time_series(output,surface_variables.index(var_name))
    #plt.plot(ps,label='pred')
    #plt.plot(bs,label='actual')
    #plt.title(f'global mean of {var_name}')
    #plt.legend()
    #plt.savefig(f'{work}/{var_name}_{cfg.experiment.name}_plot')    

    #var_name = 'gpp'
    #ps,bs = get_time_series(output,surface_variables.index(var_name))
    #plt.plot(ps,label='pred')
    #plt.plot(bs,label='actual')
    #plt.title(f'global mean of {var_name}')
    #plt.legend()
    #plt.savefig(f'{work}/{var_name}_{cfg.experiment.name}_plot')
    
if __name__ == "__main__":
    main()