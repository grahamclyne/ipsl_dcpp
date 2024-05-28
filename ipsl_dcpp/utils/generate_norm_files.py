import torch 
from ipsl_dcpp.model.ipsl_dataset import IPSL_DCPP
from hydra import compose, initialize
import numpy as np
with initialize(version_base=None, config_path="../conf"):
    cfg = compose(config_name="config")
train = IPSL_DCPP(
    'train',
    generate_statistics=True,
    lead_time_months=1,
    surface_variables=cfg.experiment.surface_variables,
    depth_variables=cfg.experiment.depth_variables,
    delta=False,
    normalization=''
)
train_dataloader = torch.utils.data.DataLoader(train,batch_size=10,shuffle=True,)



def z_normalize(data:np.ndarray,axes:tuple):
    mean = np.nanmean(data,axis=axes,keepdims=True)
    std_dev = np.nanstd(data,axis=axes,keepdims=True)
    # mean = np.broadcast_to(mean,data.shape)
    # std_dev = np.broadcast_to(std_dev,data.shape)
    # standardized_data = (data - mean) / std_dev
    return mean,std_dev

surface_means = []
#plev_means = []
depth_means =[]

surface_stds = []
#plev_stds = []
depth_stds = []

# datas = []
for count in range(100):
    # x,y,t1,t2 = batch
    batch = next(iter(train_dataloader))
  #  print(surface.shape)
    surface = batch['state_surface'].squeeze()
    #plev = batch['state_level'].squeeze()
    depth = batch['state_depth'].squeeze()
    # datas.append(x)
    print(count)
   # print(surface.shape)
  #  i_mean,i_std = z_normalize(surface.numpy(),(1,2))
    i_mean,i_std = z_normalize(surface.numpy(),0)

   # t_mean,t_std = z_normalize(plev.numpy(),(1,2,3))
    #d_mean,d_std = z_normalize(depth.numpy(),(1,2,3))
    d_mean,d_std = z_normalize(depth.numpy(),0)
    surface_means.append(i_mean)
    surface_stds.append(i_std)
   # plev_means.append(t_mean)
   # plev_stds.append(t_std)
    depth_means.append(d_mean)
    depth_stds.append(d_std)
surface_means_out = np.nanmean(np.stack(surface_means),axis=0)
surface_stds_out = np.nanmean(np.stack(surface_stds),axis=0)
#plev_means_out = np.nanmean(np.array(plev_means),axis=0)
#plev_stds_out = np.nanmean(np.array(plev_stds),axis=0)
depth_means_out = np.nanmean(np.stack(depth_means),axis=0)
depth_stds_out = np.nanmean(np.stack(depth_stds),axis=0)

np.save('data/spatial_multi_var_surface_means',surface_means_out)
np.save('data/spatial_multi_var_surface_stds',surface_stds_out)
#np.save('data/plev_means',plev_means_out)
#np.save('data/plev_stds',plev_stds_out)
np.save('data/spatial_depth_means',depth_means_out)
np.save('data/spatial_depth_stds',depth_stds_out)


#def pooled_standard_deviation(std_devs, sample_sizes):
#    # Check if the lengths of std_devs and sample_sizes are the same
#    if len(std_devs) != len(sample_sizes):
#        raise ValueError("The lengths of std_devs and sample_sizes must be the same")

    # Calculate the pooled standard deviation
#    numerator = np.sum((sample_sizes - 1) * np.square(std_devs))
#    denominator = np.sum(sample_sizes - 1)
#    pooled_std_dev = np.sqrt(numerator / denominator)

#    return pooled_std_dev