# import os 
from ipsl_dcpp.model.ipsl_dataset import IPSL_DCPP
import lightning.pytorch as pl
import torch
import hydra
from hydra import compose, initialize
import subprocess
import os
import pickle

with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config",overrides=["experiment=one_month_multiple_variable_v100"])
scratch = os.environ['SCRATCH']
work = os.environ['WORK']


def compute_predict(run_id,delta,normalization,predict_length,var_index):
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
    trainer = pl.Trainer(fast_dev_run=predict_length)
    output = trainer.predict(model, dataloader)
    predictions_out = []
    batch_out = []
    for index in range(0,len(output)):
        pred = output[index][-2]
        batch = output[index][-1]
        pred,batch = test.denormalize(pred,batch)
        predictions = pred['next_state_surface'].squeeze()[var_index]
        batch_data = batch['next_state_surface'].squeeze()[var_index]
      #  predictions = pred['next_state_surface'].squeeze()[:,var_index]
      #  batch_data = batch['next_state_surface'].squeeze()[:,var_index]
    predictions_out.append(predictions)
    batch_out.append(batch_data)
    return (predictions_out,batch_out)
  #  lat_coeffs_equi = torch.tensor([torch.cos(x) for x in torch.arange(-torch.pi/2, torch.pi/2, torch.pi/143)])
  #  lat_coeffs =  (lat_coeffs_equi/lat_coeffs_equi.mean())[None, None, None, :, None]
  #  f, axs = plt.subplots(3, 4, figsize=(16, 6))
  #  axs = axs.flatten()
  #  ds = xr.open_dataset(test.files[0])
  #  shell = ds.isel(time=0)

  #  subset_lats = lat_coeffs[:,:,:,85:110,:]
  #  subset_lats = lat_coeffs
  #  sub_shell = shell.isel(lat=slice(85,110),lon=slice(25,50))
  #  sub_shell = shell
  #  for index in range(0,len(output),3):
  #      pred = output[index][-2]
  #      batch = output[index][-1]
  #      pred,batch = test.denormalize(pred,batch)
  #      predictions = pred['next_state_surface'].squeeze()[var_index]
  #      batch_data = batch['next_state_surface'].squeeze()[var_index]
     #   predictions = pred['next_state_surface'].squeeze()[:,var_index]
     #   batch_data = batch['next_state_surface'].squeeze()[:,var_index]
        #sub_shell[surface_var_name]['time'].data = next(iter(test_dataloader))['time'][0]
        #sub_shell[surface_var_name].data = predictions
        # xx = shell[surface_var_name].plot.pcolormesh(ax=axs[index//3],cmap='ocean',add_colorbar=False,add_labels=False)
      #  pred_ax = axs[index//3].pcolormesh(predictions,cmap='ocean',vmin=0,vmax=torch.max(batch_data))
  #      pred_ax = axs[index//3].pcolormesh(predictions,cmap='ocean')        
  #      axs[index//3].set_title(f'2014-{index+1}')
        #sub_shell[surface_var_name].data = batch_data
        #shell[surface_var_name].plot.pcolormesh(ax=axs[(index//3) + 4],cmap='ocean',add_colorbar=False,add_labels=False)
        #batch_ax = axs[index//3+4].pcolormesh(batch_data ,cmap='ocean',vmin=0,vmax=torch.max(batch_data))
   #     batch_ax = axs[index//3+4].pcolormesh(batch_data ,cmap='ocean')


    #    rmse = torch.sqrt((predictions - batch_data).pow(2).mul(subset_lats))
        #sub_shell[surface_var_name].data = mse.squeeze()
        #shell[surface_var_name].plot.pcolormesh(ax=axs[(index//3) + 8],cmap='ocean',vmax=1e7,vmin=0)
    #    rmse_ax = axs[(index//3) + 8].pcolormesh(rmse.squeeze(),cmap='ocean',vmin=0)
    #    axs[(index//3) + 8].set_xlabel('')
    #    axs[(index//3) + 8].set_title('')
    #    axs[(index//3) + 8].set_ylabel('')
    #f.colorbar(pred_ax, ax=axs[index//3],cmap='ocean',location='right')
    #f.colorbar(batch_ax, ax=axs[index//3+4],cmap='ocean',location='right')
    #f.colorbar(rmse_ax, ax=axs[index//3+8],cmap='ocean',location='right')
    # rollout_data = rollout(rollout_length,dataloader,model)
    # denormalized_data = [dataloader.dataset.denormalize(rollout_data[0][i],rollout_data[1][i]) for i in range(rollout_length)]
    # denorm_pred = [x[0] for x in denormalized_data]
    # denorm_batch = [x[1] for x in denormalized_data]
    # print(len(rollout_data[0]))
    # if(normalization == 'normal'):
    #     climatology_surface_means = np.load('data/climatology_surface_means.npy')
    #     climatology_surface_stds = np.broadcast_to(np.expand_dims(np.load(f'data/climatology_surface_stds.npy'),(-2,-1)),(12,91,143,144))
    #     pred_climatology_normalized = []
    #     batch_climatology_normalized = []
    #     for i in range(rollout_length):
    #         pred_climatology_normalized.append((denormalized_data[i][0]['next_state_surface'] - climatology_surface_means[(i+1) % 12]) / climatology_surface_stds[(i+1) % 12])
    #         batch_climatology_normalized.append((denormalized_data[i][1]['next_state_surface'] - climatology_surface_means[(i+1) % 12]) / climatology_surface_stds[(i+1) % 12])    
    # else:
    #     pred_climatology_normalized = [x['next_state_surface'] for x in rollout_data[0]]
    #     batch_climatology_normalized = [x['next_state_surface'] for x in rollout_data[1]]
    return f


rollout_length = 36
predictions_gpp = [compute_predict('6a0adc60',False,'normal',rollout_length,89),
            compute_predict('4c605229',False,'climatology',rollout_length,89),
            compute_predict('2ab31632',False,'spatial_normal',rollout_length,89),
            compute_predict('153d9428',True,'normal',rollout_length,89),
            compute_predict('9cbb1f05',True,'climatology',rollout_length,89),
            compute_predict('eb38bfdd',True,'spatial_normal',rollout_length,89),
           ]

predictions_tas = [compute_predict('6a0adc60',False,'normal',rollout_length,50),
            compute_predict('4c605229',False,'climatology',rollout_length,50),
            compute_predict('2ab31632',False,'spatial_normal',rollout_length,50),
            compute_predict('153d9428',True,'normal',rollout_length,50),
            compute_predict('9cbb1f05',True,'climatology',rollout_length,50),
            compute_predict('eb38bfdd',True,'spatial_normal',rollout_length,50),

           ]

with open('gpp_predictions.pickle', 'wb') as handle:
    pickle.dump(predictions_gpp, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('tas_predictions.pickle', 'wb') as handle:
    pickle.dump(predictions_tas, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
    