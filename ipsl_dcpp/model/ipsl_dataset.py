import glob
import os 
from tqdm import tqdm
import xarray as xr
import torch
import numpy as np
import pickle

torch.set_default_dtype(torch.float32)

class IPSL_DCPP(torch.utils.data.Dataset):
    def __init__(self,
                 domain,
                 lead_time_months,
                 generate_statistics,
                 surface_variables,
                 depth_variables,
                 plev_variables,
                 delta,
                 normalization,
                 work_path,
                 scratch_path
                ):
        self.work = work_path
        self.scratch = scratch_path
        self.delta = delta
        self.surface_variables=surface_variables
        self.depth_variables=depth_variables
        self.plev_variables=plev_variables

        self.domain = domain
        #self.files = list(glob.glob(f'{self.scratch}/*_1.nc'))
        self.files = list(glob.glob(f'{self.scratch}/*.nc'))
        self.normalization = normalization
        self.var_mask = torch.from_numpy(np.load(f'{self.work}/data/land_mask.npy'))
   #     self.files = list(glob.glob(f'{self.scratch}/1970*.nc'))
        
       #by year
        # self.files = dict(
       #             all_=[str(x) for x in self.files],
       #             train=[str(x) for x in self.files if any(substring in x for substring in [str(x) for x in list(range(1960,1965))])],
       #             val = [str(x) for x in self.files if any(substring in x for substring in [str(x) for x in list(range(2000,2005))])],
       #             test = [str(x) for x in self.files if any(substring in x for substring in [str(x) for x in list(range(2012,2014))])])[domain]
        self.files = dict(
                    all_=[str(x) for x in self.files],
                    train= [str(x) for x in  self.files if any(substring in x for substring in ["_" + str(x) + '.nc'  for x in range(1,8)])],
                    val =  [str(x) for x in  self.files if any(substring in x for substring in ["_" + str(x) + '.nc'  for x in range(8,10)])],
                    test =  [str(x) for x in  self.files if any(substring in x for substring in ["_" + str(x) + '.nc'  for x in range(10,11)])])[domain]
       # [str(x) for x in files if any(substring in x for substring in ["_" + str(x) + '.nc'  for x in range(1,8)])]
        self.nfiles = len(self.files)
        self.xr_options = dict(engine='netcdf4', cache=True)
        self.lead_time_months = lead_time_months
        self.land_mask = np.expand_dims(np.load(f'{self.work}/data/land_mask.npy'),axis=(0,1))
       # self.surface_means = np.expand_dims(np.load(f'{self.work}/data/single_var_surface_means.npy'),axis=(0,1))
       # self.surface_stds = np.expand_dims(np.load(f'{self.work}/data/single_var_surface_stds.npy'),axis=(0,1))
     #   self.surface_means = np.broadcast_to(np.expand_dims(np.nanmean(np.load(f'{self.work}/data/climatology_surface_means.npy'),axis=(-1,-2,-4)),(-1,-2)),(12,91,143,144))
     #   self.surface_stds = np.broadcast_to(np.expand_dims(np.nanmean(np.load(f'{self.work}/data/climatology_surface_stds.npy'),axis=(-1,-2,-4)),(-1,-2)),(12,91,143,144))
     #   self.depth_means = np.broadcast_to(np.expand_dims(np.nanmean(np.load(f'{self.work}/data/climatology_depth_means.npy'),axis=(-1,-2,-5)),(-1,-2)),(12,3,11,143,144))
     #   self.depth_stds = np.broadcast_to(np.expand_dims(np.nanmean(np.load(f'{self.work}/data/climatology_depth_stds.npy'),axis=(-1,-2,-5)),(-1,-2)),(12,3,11,143,144))
        
        
        if(self.normalization == 'climatology'):
            self.surface_means = np.load(f'{self.work}/data/climatology_surface_means.npy')
            self.surface_stds = np.broadcast_to(np.expand_dims(np.load(f'{self.work}/data/climatology_surface_stds.npy'),(-2,-1)),(12,91,143,144))
            self.depth_means = np.load(f'{self.work}/data/climatology_depth_means.npy')
           # self.depth_stds = np.load(f'{self.work}/data/climatology_depth_stds.npy')[:,0]
            self.depth_stds = np.broadcast_to(np.expand_dims(np.load(f'{self.work}/data/climatology_depth_stds.npy'),(-2,-1)),(12,3,11,143,144))
           # self.plev_means = np.load(f'{self.work}/data/climatology_plev_means.npy')
           # self.depth_stds = np.load(f'{self.work}/data/climatology_depth_stds.npy')[:,0]
            #self.plev_stds = np.broadcast_to(np.expand_dims(np.load(f'{self.work}/data/climatology_plev_stds.npy'),(-2,-1)),(12,8,19,143,144))
        elif (self.normalization == 'normal'):
            self.surface_means = np.broadcast_to(np.expand_dims(np.load(f'{self.work}/data/surface_means.npy'),(-2,-1)),(91,143,144))
            self.surface_stds = np.broadcast_to(np.expand_dims(np.load(f'{self.work}/data/surface_stds.npy'),(-2,-1)),(91,143,144))
            self.depth_means = np.broadcast_to(np.expand_dims(np.load(f'{self.work}/data/depth_means.npy'),(-3,-2,-1)),(3,11,143,144))
            self.depth_stds = np.broadcast_to(np.expand_dims(np.load(f'{self.work}/data/depth_stds.npy'),(-3,-2,-1)),(3,11,143,144))         
        elif(self.normalization == 'spatial_normal'):
            self.surface_means = np.load(f'{self.work}/data/spatial_multi_var_surface_means.npy').squeeze()
            self.surface_stds = np.nanmean(np.load(f'{self.work}/data/spatial_multi_var_surface_stds.npy').squeeze(),axis=(-2,-1),keepdims=True)
            self.depth_means = np.load(f'{self.work}/data/spatial_depth_means.npy').squeeze()
            self.depth_stds = np.nanmean(np.load(f'{self.work}/data/spatial_depth_stds.npy').squeeze(),axis=(-2,-1),keepdims=True)
        self.surface_delta_stds = np.expand_dims(np.nanmean(np.load(f'{self.work}/data/surface_delta_std.npy'),axis=(-1,-2)),axis=(0,-1,-2))
        self.depth_delta_stds = np.expand_dims(np.nanmean(np.load(f'{self.work}/data/depth_delta_std.npy'),axis=(-1,-2)),axis=(0,-1,-2))
      #  self.plev_delta_stds = np.expand_dims(np.nanmean(np.load(f'{self.work}/data/plev_delta_std.npy'),axis=(-1,-2)),axis=(0,-1,-2))
        self.plev_delta_stds = np.ones([8,19,143,144])
        self.atmos_forcings = np.load(f'{self.work}/data/atmos_forcings.npy')
        self.atmos_forcings = (self.atmos_forcings - self.atmos_forcings.mean(axis=1,keepdims=True)) / self.atmos_forcings.std(axis=1,keepdims=True)
        #self.plev_means = np.expand_dims(np.load(f'{self.work}/ipsl_dcpp/data/plev_means.npy'),axis=(1,2,3))
        #self.plev_stds = np.expand_dims(np.load(f'{self.work}/ipsl_dcpp/data/plev_stds.npy'),axis=(1,2,3))
        self.plev_means = np.zeros([8,19,143,144])
        self.plev_stds = np.ones([8,19,143,144])
        self.generate_statistics=generate_statistics
        self.timestamps = []
        self.id2pt_path = f'{self.work}/data/{domain}_id2pt.pkl'
        if os.path.exists(self.id2pt_path):
            with open(self.id2pt_path, 'rb') as handle:
                self.id2pt = pickle.load(handle)
        else:
            for fid, f in tqdm(enumerate(self.files)):
                with xr.open_dataset(f, **self.xr_options) as obs:
                    var_id = f.split('.')[0][-1]
                    file_stamps = [(fid, i, t,var_id) for (i, t) in enumerate(obs.time.to_numpy())]
                    #if doing autoregressive - don't include the last -leadtime- amount of each timeseries to avoid indexing overflow issues
                    if(self.lead_time_months == 0):
                        self.timestamps.extend(file_stamps)
                    else:
                        self.timestamps.extend(file_stamps[:-(self.lead_time_months)])
                    #self.timestamps.extend(file_stamps)
            self.timestamps = sorted(self.timestamps, key=lambda x: x[-1]) # sort by timestamp
            self.id2pt = {i:(file_id, line_id) for (i, (file_id, line_id, var_id,s)) in enumerate(self.timestamps)}
            with open(self.id2pt_path, 'wb') as handle:
                pickle.dump(self.id2pt,handle)
                
    def __len__(self):
        return len(self.id2pt) 
    
    def xarr_to_tensor(self, obsi,variables):
        if(len(variables) == 0):
            return []
        data_np = obsi[variables].to_array().to_numpy()
        data_th = torch.from_numpy(data_np)
        return data_th
    
    def preprocess(self, clim,clim_next):
        out = dict()
        input_surface_variables = self.xarr_to_tensor(clim, list(self.surface_variables))
        input_plev_variables = self.xarr_to_tensor(clim,list(self.plev_variables))
        input_depth_variables = self.xarr_to_tensor(clim,list(self.depth_variables))
        

        target_surface_variables = self.xarr_to_tensor(clim_next, list(self.surface_variables))
        target_plev_variables = self.xarr_to_tensor(clim_next,list(self.plev_variables))
        target_depth_variables = self.xarr_to_tensor(clim_next,list(self.depth_variables))
        time = clim.time.dt.strftime('%Y-%m').item()
        next_time = clim_next.time.dt.strftime('%Y-%m').item()
        cur_month_index = int(time.split('-')[-1]) - 1
        next_month_index = int(next_time.split('-')[-1]) - 1 
        cur_year_index = int(time.split('-')[0]) - 1960
        next_year_index = int(next_time.split('-')[0]) - 1960
        #cur_year_forcings = np.broadcast_to(np.expand_dims(self.atmos_forcings[:,cur_year_index],(1,2)),(4,143,144))
        if(not self.generate_statistics):
            if(self.normalization == 'climatology'):
                input_surface_variables = (input_surface_variables - self.surface_means[cur_month_index]) / (self.surface_stds[cur_month_index])
                input_depth_variables = (input_depth_variables - self.depth_means[cur_month_index]) / (self.depth_stds[cur_month_index])
               # input_plev_variables = (input_plev_variables - self.plev_means[cur_month_index]) / (self.plev_stds[cur_month_index])

                target_surface_variables = (target_surface_variables - self.surface_means[next_month_index]) / (self.surface_stds[next_month_index])
                target_depth_variables = (target_depth_variables - self.depth_means[next_month_index]) / (self.depth_stds[next_month_index])
               # target_plev_variables = (target_plev_variables - self.plev_means[cur_month_index]) / (self.plev_stds[cur_month_index])

            elif(self.normalization == 'normal' or self.normalization == 'spatial_normal'):
                input_surface_variables = (input_surface_variables - self.surface_means) / self.surface_stds
                input_depth_variables = (input_depth_variables - self.depth_means) / self.depth_stds

                target_surface_variables = (target_surface_variables - self.surface_means) / self.surface_stds
                target_depth_variables = (target_depth_variables - self.depth_means) / self.depth_stds
            if(self.delta):
           # surface_mask = (self.surface_delta_stds != 0)
           # depth_mask = (self.depth_delta_stds != 0)
               #target_surface_variables = (target_surface_variables[surface_mask] - input_surface_variables[surface_mask]) / (self.surface_delta_stds[surface_mask])
            #target_depth_variables = (target_depth_variables[depth_mask] - input_depth_variables[depth_mask]) / (self.depth_delta_stds[depth_mask])
                target_plev_variables = (target_plev_variables - input_plev_variables) / self.plev_delta_stds
                target_surface_variables = (target_surface_variables - input_surface_variables) / self.surface_delta_stds
                target_depth_variables = (target_depth_variables - input_depth_variables) / self.depth_delta_stds
            input_surface_variables = np.expand_dims(np.nan_to_num(input_surface_variables,0),0)
            input_plev_variables = np.expand_dims(np.nan_to_num(input_plev_variables,0),0)
            input_depth_variables = np.expand_dims(np.nan_to_num(input_depth_variables,0),0)
            target_surface_variables = np.expand_dims(np.nan_to_num(target_surface_variables,0),0)
            target_plev_variables = np.expand_dims(np.nan_to_num(target_plev_variables,0),0)
            target_depth_variables = np.expand_dims(np.nan_to_num(target_depth_variables,0),0)
        out.update(dict(
                    # state_surface=input_surface_variables.astype(np.float32),
                    # state_level=input_plev_variables.astype(np.float32),
                    # state_depth=input_depth_variables.astype(np.float32),
                    # state_constant=self.land_mask.astype(np.float32),
                    # next_state_surface=target_surface_variables.astype(np.float32),
                    # next_state_level=target_plev_variables.astype(np.float32),
                    # next_state_depth=target_depth_variables.astype(np.float32),
                    state_surface=input_surface_variables,
                    state_level=input_plev_variables,
                    state_depth=input_depth_variables,
                    state_constant=self.land_mask,
                    next_state_surface=target_surface_variables,
                    next_state_level=target_plev_variables,
                    next_state_depth=target_depth_variables,
                    time=time,
                    next_time=next_time,
           # forcings=cur_year_forcings
                ))


        return out

    
    def denormalize(self, pred,batch):
        
        device = batch['next_state_surface'].device
        cur_month = int(batch['time'][0].split('-')[-1]) - 1     
        next_month = int(batch['next_time'][0].split('-')[-1]) - 1    
        #   denorm_level = lambda x: x.to(device)*torch.from_numpy(self.plev_stds).to(device) + torch.from_numpy(self.plev_means).to(device)
    #    if(self.delta):
    #        if(pred != None):
    #            pred['next_state_surface'] = pred['next_state_surface']*self.surface_delta_stds + batch['state_surface']
    #            pred['next_state_depth'] = pred['next_state_depth']*self.depth_delta_stds + batch['state_depth']
    #        batch['next_state_surface'] = batch['next_state_surface']*self.surface_delta_stds + batch['state_surface']
    #        batch['next_state_depth'] = batch['next_state_depth']*self.depth_delta_stds + batch['state_depth']
        if(self.normalization == 'climatology'):
            denorm_surface = lambda x,month_index: x.to(device)*torch.from_numpy(self.surface_stds[month_index]).to(device) + torch.from_numpy(self.surface_means[month_index]).to(device)
            denorm_depth = lambda x,month_index: x.to(device)*torch.from_numpy(self.depth_stds[month_index]).to(device) + torch.from_numpy(self.depth_means[month_index]).to(device)
            
        elif(self.normalization == 'normal' or self.normalization == 'spatial_normal'):
            denorm_surface = lambda x,month_index: x.squeeze().to(device)*torch.from_numpy(self.surface_stds).to(device) + torch.from_numpy(self.surface_means).to(device)
            denorm_depth = lambda x,month_index: x.squeeze().to(device)*torch.from_numpy(self.depth_stds).to(device) + torch.from_numpy(self.depth_means).to(device)    
            
        if(pred != None):
            pred = dict(#next_state_level=denorm_level(pred['next_state_level']),
                        next_state_surface=torch.where(self.var_mask,denorm_surface(pred['next_state_surface'],next_month),torch.nan),
                        next_state_depth=denorm_depth(pred['next_state_depth'],next_month),
            )

        batch = dict(#next_state_level=denorm_level(batch['next_state_level']),
                    next_state_surface=torch.where(self.var_mask,denorm_surface(batch['next_state_surface'],next_month),torch.nan),
                     next_state_depth=denorm_depth(batch['next_state_depth'],next_month),
                      state_depth=denorm_depth(batch['state_depth'],next_month),
                    state_constant=batch['state_constant'],
                    state_surface=torch.where(self.var_mask==1,denorm_surface(batch['state_surface'],cur_month),torch.nan),
                    time=batch['time'],
        next_time=batch['next_time'])

        return pred, batch

    
    
    
    def __getitem__(self, i):
        file_id, line_id = self.id2pt[i]
        next_line_id = line_id + self.lead_time_months
        obs = xr.open_dataset(self.files[file_id], **self.xr_options)
        clim = obs.isel(time=line_id)
        clim_next = obs.isel(time=next_line_id)
        
        out = self.preprocess(clim,clim_next)
        obs.close()
        return out
