plev_variables = ['hur','hus','o3','ta','ua','va','wap','zg']
import glob
import os 
from tqdm import tqdm
import xarray as xr
import torch
import numpy as np
import pickle
work = os.environ['WORK'] + '/ipsl_dcpp/ipsl_dcpp'
scratch = os.environ['SCRATCH']

class IPSL_DCPP(torch.utils.data.Dataset):
    def __init__(self,
                 domain,
                 lead_time_months,
                 generate_statistics,
                 surface_variables,
                 depth_variables,
                 delta
                ):
        self.delta = delta
        self.surface_variables=surface_variables
        self.depth_variables=depth_variables
        self.domain = domain
        #self.files = list(glob.glob(f'{scratch}/*_1.nc'))
        self.files = list(glob.glob(f'{scratch}/*.nc'))
        self.files = dict(
                    all_=[str(x) for x in self.files],
                    train=[str(x) for x in self.files if any(substring in x for substring in [str(x) for x in list(range(1960,1965))])],
                    val = [str(x) for x in self.files if any(substring in x for substring in [str(x) for x in list(range(2000,2005))])],
                    test = [str(x) for x in self.files if any(substring in x for substring in [str(x) for x in list(range(2012,2014))])])[domain]
        self.nfiles = len(self.files)
        self.xr_options = dict(engine='netcdf4', cache=True)
        self.lead_time_months = lead_time_months
        self.land_mask = np.expand_dims(np.load(f'{work}/data/land_mask.npy'),axis=(0,1))
       # self.surface_means = np.expand_dims(np.load(f'{work}/data/single_var_surface_means.npy'),axis=(0,1))
       # self.surface_stds = np.expand_dims(np.load(f'{work}/data/single_var_surface_stds.npy'),axis=(0,1))
     #   self.surface_means = np.broadcast_to(np.expand_dims(np.nanmean(np.load(f'{work}/data/climatology_surface_means.npy'),axis=(-1,-2,-4)),(-1,-2)),(12,91,143,144))
     #   self.surface_stds = np.broadcast_to(np.expand_dims(np.nanmean(np.load(f'{work}/data/climatology_surface_stds.npy'),axis=(-1,-2,-4)),(-1,-2)),(12,91,143,144))
     #   self.depth_means = np.broadcast_to(np.expand_dims(np.nanmean(np.load(f'{work}/data/climatology_depth_means.npy'),axis=(-1,-2,-5)),(-1,-2)),(12,3,11,143,144))
     #   self.depth_stds = np.broadcast_to(np.expand_dims(np.nanmean(np.load(f'{work}/data/climatology_depth_stds.npy'),axis=(-1,-2,-5)),(-1,-2)),(12,3,11,143,144))
        
        
        
        self.surface_means = np.load(f'{work}/data/climatology_surface_means.npy')[:,0]
        self.surface_stds = np.load(f'{work}/data/climatology_surface_stds.npy')[:,0]
        self.depth_means = np.load(f'{work}/data/climatology_depth_means.npy')[:,0]
        self.depth_stds = np.load(f'{work}/data/climatology_depth_stds.npy')[:,0]
        self.surface_delta_stds = np.expand_dims(np.load(f'{work}/data/surface_delta_std.npy'),axis=0)
        self.depth_delta_stds = np.expand_dims(np.load(f'{work}/data/depth_delta_std.npy'),axis=0)
        self.atmos_forcings = np.load(f'{work}/data/atmos_forcings.npy')
       # print(self.surface_means.shape)
       # print(self.depth_means.shape) 
       # self.plev_means = np.expand_dims(np.load(f'{work}/ipsl_dcpp/data/plev_means.npy'),axis=(1,2,3))
       # self.plev_stds = np.expand_dims(np.load(f'{work}/ipsl_dcpp/data/plev_stds.npy'),axis=(1,2,3))
        self.generate_statistics=generate_statistics
        self.timestamps = []
        self.id2pt_path = f'{work}/data/{domain}_id2pt.pkl'
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
       # input_plev_variables = self.xarr_to_tensor(clim,plev_variables)
        input_depth_variables = self.xarr_to_tensor(clim,list(self.depth_variables))
        

        target_surface_variables = self.xarr_to_tensor(clim_next, list(self.surface_variables))
        #target_plev_variables = self.xarr_to_tensor(clim_next,plev_variables)
        target_depth_variables = self.xarr_to_tensor(clim_next,list(self.depth_variables))
        time = clim.time.dt.strftime('%Y-%m').item()
        next_time = clim_next.time.dt.strftime('%Y-%m').item()
        cur_month_index = int(time.split('-')[-1]) - 1
        next_month_index = int(next_time.split('-')[-1]) - 1 
        cur_year_index = int(time.split('-')[0]) - 1960
        next_year_index = int(next_time.split('-')[0]) - 1960
       # cur_year_forcings = np.broadcast_to(np.expand_dims(self.atmos_forcings[:,cur_year_index],(1,2,3)),(4,12,144,143))
        if(not self.generate_statistics):
        #    print(input_surface_variables.shape)
          #  print('before normalize',input_surface_variables[31][833//143][-31])
            #print(self.surface_means[cur_month_index].shape)
           # print(self.surface_means[cur_month_index][0][31][833//143][-31])
           # print(self.surface_stds[cur_month_index][31][833//143][-31])
          #  print(self.surface_variables[31])
      #      print(self.surface_means.shape)
            eps = 1
            #eps = 0
            #print(input_surface_variables.shape)
            input_surface_variables = (input_surface_variables - self.surface_means[cur_month_index]) / (self.surface_stds[cur_month_index] + eps)
        #    input_plev_variables = (input_plev_variables - self.plev_means) / self.plev_stds
            input_depth_variables = (input_depth_variables - self.depth_means[cur_month_index]) / (self.depth_stds[cur_month_index] + eps)
            target_surface_variables = (target_surface_variables - self.surface_means[next_month_index]) / (self.surface_stds[next_month_index] + eps)
        #    target_plev_variables = (target_plev_variables - self.plev_means) / self.plev_stds
            target_depth_variables = (target_depth_variables - self.depth_means[next_month_index]) / (self.depth_stds[next_month_index] + eps)

            input_surface_variables = np.expand_dims(np.nan_to_num(input_surface_variables,0),0)
         #   input_plev_variables = np.nan_to_num(input_plev_variables,0)
            input_depth_variables = np.expand_dims(np.nan_to_num(input_depth_variables,0),0)
            target_surface_variables = np.expand_dims(np.nan_to_num(target_surface_variables,0),0)
          #  target_plev_variables = np.nan_to_num(target_plev_variables,0)
            target_depth_variables = np.expand_dims(np.nan_to_num(target_depth_variables,0),0)
            #assert sum(np.isnan(out['state_depth']).flatten()) == 0
            #assert sum(np.isnan(out['state_surface']).flatten()) == 0
         #   print(input_surface_variables.shape)
         #   print(input_surface_variables[0][31][833//143][-31])
      #  print(self.depth_delta_stds.shape)
      #  print(input_depth_variables.shape)
      #  print(target_depth_variables.shape)
      #  print(target_depth_variables)
        #print(sum(np.isnan(target_surface_variables).flatten()))
            if(self.delta):
           # surface_mask = (self.surface_delta_stds != 0)
           # depth_mask = (self.depth_delta_stds != 0)
               #target_surface_variables = (target_surface_variables[surface_mask] - input_surface_variables[surface_mask]) / (self.surface_delta_stds[surface_mask])
           #target_depth_variables = (target_depth_variables[depth_mask] - input_depth_variables[depth_mask]) / (self.depth_delta_stds[depth_mask])
          #  print('before delta adjust',target_surface_variables)
          #  print(len(np.nonzero(np.isnan(target_surface_variables))[0]))
                target_surface_variables = (target_surface_variables - input_surface_variables) / (self.surface_delta_stds + eps)
                target_depth_variables = (target_depth_variables - input_depth_variables) / (self.depth_delta_stds + eps)
                #target_depth_variables[target_depth_variables != target_depth_variables] = 0
                #target_surface_variables[target_surface_variables != target_surface_variables] = 0
                target_surface_variables = np.nan_to_num(target_surface_variables,0)
                target_depth_variables = np.nan_to_num(target_depth_variables,0)
              #  print('after_delta adjust',target_surface_variables)
              #  print('after_delta adjust',target_depth_variables)

               # print(len(np.nonzero(np.isnan(target_surface_variables))[0]))
               # print(len(np.nonzero(np.isnan(target_depth_variables))[0]))
               # print(target_surface_variables.mean())

        #print(sum(np.isnan(target_surface_variables).flatten()))
     #   print('tar surf',target_surface_variables)
      #  print('tar_depth',target_depth_variables)
 
        out.update(dict(
                    state_surface=input_surface_variables,
           #         state_level=input_plev_variables,
                    state_depth=input_depth_variables,
                    state_constant=self.land_mask,
                    next_state_surface=target_surface_variables,
            #        next_state_level=target_plev_variables,
                    next_state_depth=target_depth_variables,
                    time=time,
                    next_time=next_time,
           # forcings=cur_year_forcings
                ))


        return out

    
    def denormalize(self, pred, batch):
        device = pred['next_state_surface'].device
     #   denorm_level = lambda x: x.to(device)*torch.from_numpy(self.plev_stds).to(device) + torch.from_numpy(self.plev_means).to(device)
        denorm_surface = lambda x: x.to(device)*torch.from_numpy(self.surface_stds).to(device) + torch.from_numpy(self.surface_means).to(device)
        denorm_depth = lambda x: x.to(device)*torch.from_numpy(self.depth_stds).to(device) + torch.from_numpy(self.depth_means).to(device)

        pred = dict(#next_state_level=denorm_level(pred['next_state_level']),
                    next_state_surface=denorm_surface(pred['next_state_surface']),
                    next_state_depth=denorm_depth(pred['next_state_depth']))

        batch = dict(#next_state_level=denorm_level(batch['next_state_level']),
                    next_state_surface=denorm_surface(batch['next_state_surface']),
                     next_state_depth=denorm_depth(batch['next_state_depth']),
                    state_surface=denorm_surface(batch['state_surface']))

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
