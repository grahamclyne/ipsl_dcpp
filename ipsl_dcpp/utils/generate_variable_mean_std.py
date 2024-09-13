import torch
import numpy as np
from hydra import compose, initialize
import hydra
with initialize(version_base=None, config_path="../conf"):
    cfg = compose(config_name="config")

#year = 1960
#variation = 1
#Amon =  xr.open_mfdataset(f'{store_dir}/s{year}-r{variation}i1p1f1/Amon/*.nc',compat='minimal')
#get climatology over train period for a year period
#train = IPSL_DCPP('train',generate_statistics=True,lead_time_months=1)
# train =  IPSL_DCPP('train',0,generate_statistics=True,
#               delta=False,
#               surface_variables=cfg.module.surface_variables,
#               depth_variables=cfg.module.depth_variables,
#                   plev_variables=cfg.module.plev_variables,
#                   normalization='climatology',
#                       work_path=cfg.cluster.work_path,
#     scratch_path=cfg.cluster.scratch_path,)

train = hydra.utils.instantiate(
    cfg.dataloader.dataset,
    domain='train',
    generate_statistics=True, 
    delta=False,
)

# train_dataloader = torch.utils.data.DataLoader(
#     train,batch_size=1,shuffle=True,num_workers=1)
# surface_means_out = []
# surface_stds_out = []
# depth_means_out = []
# depth_stds_out = []
# plev_means_out = []
# plev_stds_out = []
#iter_batch = iter(train_dataloader)
surface = []

for count in range(4000):
    sample_idx = torch.randint(len(train), size=(1,)).item()
    batch = train[sample_idx]
    surface.append(batch['state_surface'].squeeze())
    print(batch['state_surface'].shape)

surface = np.stack(surface)
surface_means = np.nanmean(surface,axis=(0,2,3))
surface_stds = np.nanstd(surface,axis=(0,2,3))
#depth_month_means = [np.nanmean(np.array(x),axis=0) for x in depth_months]
#depth_month_stds = [np.nanstd(np.array(x),axis=(0,3,4)) for x in depth_months]
# plev_month_means = [np.nanmean(np.stack(x),axis=0) for x in plev_months]
# plev_month_stds = [np.nanstd(np.stack(x),axis=(0,3,4)) for x in plev_months]

np.save('variable_surface_means_ensemble_split.npy',surface_means)
np.save('variable_surface_stds_ensemble_split.npy',surface_stds)
#np.save('climatology_depth_means.npy',np.stack(depth_month_means))
#np.save('climatology_depth_stds.npy',np.stack(depth_month_stds))
# np.save('climatology_plev_means_ensemble_split.npy',np.stack(plev_month_means))
# np.save('climatology_plev_stds_ensemble_split.npy',np.stack(plev_month_stds))