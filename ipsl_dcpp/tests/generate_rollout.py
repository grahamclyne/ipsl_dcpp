# import os 
from ipsl_dcpp.model.ipsl_dataset import IPSL_DCPP
import torch
import hydra
import numpy as np  
from hydra import compose, initialize
import subprocess
import os
from evaluation.visualization import rollout
with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config",overrides=["experiment=one_month_multiple_variable_v100"])
scratch = os.environ['SCRATCH']
work = os.environ['WORK']

def compute_rollout(run_id,delta,normalization,rollout_length,zeroes=False):
    #return unnormazlied and climatology-normalized data
    test = IPSL_DCPP('test',
                     lead_time_months=1,
                     surface_variables=cfg.experiment.surface_variables,
                     depth_variables=cfg.experiment.depth_variables,
                     generate_statistics=False,
                     delta=delta,
                     normalization=normalization
                    )
    dataloader = torch.utils.data.DataLoader(test,batch_size=1,shuffle=False,num_workers=1)
    out = subprocess.run(f'ls -tr {scratch}/checkpoint_{run_id}/ | tail -n 1',shell=True,capture_output=True,text=True)
    path = out.stdout.strip("\n")
    checkpoint_path = torch.load(f'{scratch}/checkpoint_{run_id}/{path}',map_location=torch.device('cpu'))
    model = hydra.utils.instantiate(
        cfg.experiment.module,
        backbone=hydra.utils.instantiate(cfg.experiment.backbone),
        dataset=test
    )
    model.load_state_dict(checkpoint_path['state_dict'])
    rollout_data = rollout(rollout_length,dataloader,model,zeroes)
    denormalized_data = [dataloader.dataset.denormalize(rollout_data[0][i],rollout_data[1][i]) for i in range(rollout_length)]
    denorm_pred = [x[0] for x in denormalized_data]
    denorm_batch = [x[1] for x in denormalized_data]
    print(len(rollout_data[0]))
    if(normalization == 'normal' or normalization == 'spatial_normal'):
        climatology_surface_means = np.load('data/climatology_surface_means.npy')
        climatology_surface_stds = np.broadcast_to(np.expand_dims(np.load('data/climatology_surface_stds.npy'),(-2,-1)),(12,91,143,144))
        pred_climatology_normalized = []
        batch_climatology_normalized = []
        for i in range(rollout_length):
            pred_climatology_normalized.append((denorm_pred[i]['next_state_surface'] - climatology_surface_means[(i+1) % 12]) / climatology_surface_stds[(i+1) % 12])
            batch_climatology_normalized.append((denorm_batch[i]['next_state_surface'] - climatology_surface_means[(i+1) % 12]) / climatology_surface_stds[(i+1) % 12])    
    else:
        pred_climatology_normalized = [x['next_state_surface'] for x in rollout_data[0]]
        batch_climatology_normalized = [x['next_state_surface'] for x in rollout_data[1]]
    return denorm_pred,denorm_batch,pred_climatology_normalized,batch_climatology_normalized



if __name__ == "__main__":
  
    rollout_length = 36
    rollouts = [compute_rollout('6a0adc60',False,'normal',rollout_length),
                compute_rollout('4c605229',False,'climatology',rollout_length),
                compute_rollout('153d9428',True,'normal',rollout_length),
                #compute_rollout('9cbb1f05',True,'climatology',rollout_length),
                compute_rollout('edb26faa',True,'climatology',rollout_length),
                compute_rollout('eb38bfdd',True,'spatial_normal',rollout_length),
                compute_rollout('2ab31632',False,'spatial_normal',rollout_length)
               ]

    import pickle

    with open('rollout.pickle', 'wb') as handle:
        pickle.dump(rollouts, handle, protocol=pickle.HIGHEST_PROTOCOL)
