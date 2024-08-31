
import glob
from tqdm import tqdm
import xarray as xr
import torch
import numpy as np
import pickle


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
                 data_path,
                 flattened_plev,
                 debug,
                 z_normalize,
                 mask_value,
                 plot_output_path
                ):
        self.flattened_plev = flattened_plev
        self.plot_output_path = plot_output_path
        self.mask_value = mask_value
        self.data_path = data_path
        self.delta = delta
        self.surface_variables=surface_variables
        self.depth_variables=depth_variables
        self.plev_variables=plev_variables
        self.debug = debug
        self.domain = domain
        self.z_normalize = z_normalize
        self.files = list(glob.glob(f'{self.data_path}/batch_with_tos/*.nc'))
        self.normalization = normalization
        self.elevation_data = torch.from_numpy(np.expand_dims(np.load(f'{self.data_path}/reference_data/elev_data.npy'),axis=(0)))
        self.var_mask = torch.from_numpy(np.load(f'{self.data_path}/reference_data/land_mask.npy'))
        self.plev_mask = torch.from_numpy(np.expand_dims(np.load(f'{self.data_path}/reference_data/plev_mask.npy'),(0)))
        self.ocean_mask = torch.from_numpy(np.expand_dims(np.load(f'{self.data_path}/reference_data/ocean_mask.npy'),(0)))
        lat_coeffs_equi = torch.tensor([torch.cos(x) for x in torch.arange(-torch.pi/2, torch.pi/2, torch.pi/143)])
        self.lat_coeffs_equi =  (lat_coeffs_equi/lat_coeffs_equi.mean())[None, None, None, :, None]
       #by year
        # self.files = dict(
        #           all_=[str(x) for x in self.files],
        #     #need to go to only 2004 because atmos forcings only go to 2014
        #           train=[str(x) for x in self.files if any(substring in x for substring in [str(x) for x in list(range(1960,2008))])],
        #           val = [str(x) for x in self.files if any(substring in x for substring in [str(x) for x in list(range(2009,2013))])],
        #           test = [str(x) for x in self.files if any(substring in x for substring in [str(x) for x in list(range(2013,2016))])])[domain]
        #by ensemble member
        self.files = sorted(dict(
                     all_=[str(x) for x in self.files],
                      train= [str(x) for x in  self.files if any(substring in x for substring in ["_" + str(x) + '_tos_included.nc'  for x in range(1,8)])],
                      val =  [str(x) for x in  self.files if any(substring in x for substring in ["_" + str(x) + '_tos_included.nc'  for x in range(8,11)])],
                     #train= [str(x) for x in  self.files if any(substring in x for substring in ["_" + str(x) + '_tos_included.nc'  for x in range(1,2)])],
                    # val =  [str(x) for x in  self.files if any(substring in x for substring in ["_" + str(x) + '_tos_included.nc'  for x in range(8,8)])],
                     test =  [str(x) for x in  self.files if any(substring in x for substring in ["_" + str(x) + '_tos_included.nc'  for x in range(8,11)])])[domain])
        self.nfiles = len(self.files)
        self.xr_options = dict(engine='netcdf4', cache=True)
        self.lead_time_months = lead_time_months
       # temp = xr.open_dataset(self.files[0])
        #stats_variable_subset = [list(temp.keys()).index(var) for var in surface_variables]
        #print(stats_variable_subset)
        #variable_subset = [50, 87, 88, 89, 6, 25, 13, 14, 34,-1]
        self.land_mask = torch.from_numpy(np.expand_dims(np.load(f'{self.data_path}/reference_data/land_mask.npy'),axis=(0)))
       # self.surface_means = np.expand_dims(np.load(f'{self.work}/data/single_var_surface_means.npy'),axis=(0,1))
       # self.surface_stds = np.expand_dims(np.load(f'{self.work}/data/single_var_surface_stds.npy'),axis=(0,1))
     #   self.surface_means = np.broadcast_to(np.expand_dims(np.nanmean(np.load(f'{self.work}/data/climatology_surface_means.npy'),axis=(-1,-2,-4)),(-1,-2)),(12,91,143,144))
     #   self.surface_stds = np.broadcast_to(np.expand_dims(np.nanmean(np.load(f'{self.work}/data/climatology_surface_stds.npy'),axis=(-1,-2,-4)),(-1,-2)),(12,91,143,144))
     #   self.depth_means = np.broadcast_to(np.expand_dims(np.nanmean(np.load(f'{self.work}/data/climatology_depth_means.npy'),axis=(-1,-2,-5)),(-1,-2)),(12,3,11,143,144))
     #   self.depth_stds = np.broadcast_to(np.expand_dims(np.nanmean(np.load(f'{self.work}/data/climatology_depth_stds.npy'),axis=(-1,-2,-5)),(-1,-2)),(12,3,11,143,144))
        
        if(self.normalization == 'climatology'):
            #self.surface_means = np.load(f'{self.work}/data/climatology_surface_means.npy')
            self.surface_means = np.load(f'{self.data_path}/reference_data/climatology_surface_means_ensemble_split.npy')

            self.surface_stds = np.broadcast_to(np.expand_dims(np.load(f'{self.data_path}/reference_data/climatology_surface_stds_ensemble_split.npy'),(-2,-1)),(12,len(surface_variables),143,144))
            #self.depth_means = np.load(f'{self.work}/data/climatology_depth_means.npy')
           # self.depth_stds = np.load(f'{self.work}/data/climatology_depth_stds.npy')[:,0]
            #self.depth_stds = np.broadcast_to(np.expand_dims(np.load(f'{self.work}/data/climatology_depth_stds.npy'),(-2,-1)),(12,3,11,143,144))
            self.plev_means = np.load(f'{self.data_path}/reference_data/climatology_plev_means_ensemble_split.npy')
           # self.depth_stds = np.load(f'{self.work}/data/climatology_depth_stds.npy')[:,0]
            self.plev_stds = np.broadcast_to(np.expand_dims(np.load(f'{self.data_path}/reference_data/climatology_plev_stds_ensemble_split.npy'),(-2,-1)),(12,8,3,143,144))
            if(self.z_normalize):
                self.z_means = np.expand_dims(np.nanmean(np.load(f'{self.data_path}/reference_data/after_climatology_surface_means_ensemble_split.npy'),axis=0),(-2,-1))
                self.z_stds = np.expand_dims(np.nanmean(np.load(f'{self.data_path}/reference_data/after_climatology_surface_stds_ensemble_split.npy'),axis=0),(-2,-1))
        elif (self.normalization == 'normal'):
            self.surface_means = np.broadcast_to(np.expand_dims(np.load(f'{self.data_path}/reference_data/surface_means.npy'),(-2,-1)),(9,143,144))[6]
            self.surface_stds = np.broadcast_to(np.expand_dims(np.load(f'{self.data_path}/reference_data/surface_stds.npy'),(-2,-1)),(9,143,144))[6]
          #  self.depth_means = np.broadcast_to(np.expand_dims(np.load(f'{self.work}/data/depth_means.npy'),(-3,-2,-1)),(3,11,143,144))
           # self.depth_stds = np.broadcast_to(np.expand_dims(np.load(f'{self.work}/data/depth_stds.npy'),(-3,-2,-1)),(3,11,143,144))
        elif(self.normalization == 'spatial_normal'):
            self.surface_means = np.load(f'{self.data_path}/reference_data/spatial_multi_var_surface_means.npy').squeeze()
            #print(self.surface_means.shape)
            self.surface_stds = np.nanmean(np.load(f'{self.data_path}/reference_data/spatial_multi_var_surface_stds.npy').squeeze(),axis=(-2,-1),keepdims=True)
           # print(self.surface_stds.shape)
            self.depth_means = np.load(f'{self.data_path}/reference_data/spatial_depth_means.npy').squeeze()
            self.depth_stds = np.nanmean(np.load(f'{self.data_path}/reference_data/spatial_depth_stds.npy').squeeze(),axis=(-2,-1),keepdims=True)
        
        if(self.z_normalize):
            self.surface_delta_stds = torch.Tensor(np.expand_dims(np.nanmean(np.load(f'{self.data_path}/reference_data/z_norm_surface_delta_std_ensemble_split.npy'),axis=(-1,-2)),axis=(-1,-2)))[:10]
            self.plev_delta_stds = torch.Tensor(np.expand_dims(np.nanmean(np.load(f'{self.data_path}/reference_data/z_norm_plev_delta_std_ensemble_split.npy'),axis=(-1,-2)),axis=(0,-1,-2))).reshape(8,3,1,1)
        else:
            self.surface_delta_stds = torch.Tensor(np.expand_dims(np.nanmean(np.load(f'{self.data_path}/reference_data/surface_delta_std_ensemble_split.npy'),axis=(-1,-2)),axis=(-1,-2)))[:10]
            self.plev_delta_stds = torch.Tensor(np.expand_dims(np.nanmean(np.load(f'{self.data_path}/reference_data/plev_delta_std_ensemble_split.npy'),axis=(-1,-2)),axis=(0,-1,-2))).reshape(8,3,1,1)
     #   self.depth_delta_stds = torch.Tensor(np.expand_dims(np.nanmean(np.load(f'{self.work}/data/depth_delta_std.npy'),axis=(-1,-2)),axis=(0,-1,-2)))
      #  self.plev_delta_stds = np.ones([8,19,143,144])
        # self.atmos_forcings = np.load(f'{self.work}/data/atmos_forcings.npy')
        # self.solar_forcings = np.load(f'{self.work}/data/solar_forcings.npy')
        self.atmos_forcings = np.load(f'{self.data_path}/reference_data/ssp_585_full_atmos.npy')
        self.solar_forcings = np.load(f'{self.data_path}/reference_data/full_solar.npy')
        self.atmos_forcings = (self.atmos_forcings - self.atmos_forcings.mean(axis=1,keepdims=True)) / self.atmos_forcings.std(axis=1,keepdims=True)
        self.solar_forcings = (self.solar_forcings - self.solar_forcings.mean(axis=(0,1),keepdims=True)) / self.solar_forcings.std(axis=(0,1),keepdims=True)
        #self.plev_means = np.expand_dims(np.load(f'{self.work}/ipsl_dcpp/data/plev_means.npy'),axis=(1,2,3))
        #self.plev_stds = np.expand_dims(np.load(f'{self.work}/ipsl_dcpp/data/plev_stds.npy'),axis=(1,2,3))
      #  self.plev_means = np.zeros([8,19,143,144])
      #  self.plev_stds = np.ones([8,19,143,144])
        self.generate_statistics=generate_statistics
        self.timestamps = []
        self.id2pt_path = f'{self.data_path}/reference_data/{domain}_id2pt.pkl'
        count = 0
        for fid, f in tqdm(enumerate(self.files)):
            with xr.open_dataset(f, **self.xr_options) as obs:
                var_id = f.split('.')[0][-16]
                file_stamps = [(fid, i, t,var_id) for (i, t) in enumerate(obs.time.to_numpy())]
                #if doing autoregressive - don't include the last -leadtime- amount of each timeseries to avoid indexing overflow issues and 
                if(self.lead_time_months == 0):
                    self.timestamps.extend(file_stamps)
                else:
                    self.timestamps.extend(file_stamps[:-((self.lead_time_months)+1)])
                #self.timestamps.extend(file_stamps)
            count += 1
            if(self.debug and count > 2): #just get one file
                break
     #   self.timestamps = sorted(self.timestamps, key=lambda x: x[-1]) # sort by timestamp
        self.id2pt = {i:(file_id, line_id) for (i, (file_id, line_id, var_id,s)) in enumerate(self.timestamps)}
        with open(self.id2pt_path, 'wb') as handle:
            pickle.dump(self.id2pt,handle)
                
    def __len__(self):
        return len(self.id2pt) 
    
    def xarr_to_tensor(self, obsi,variables):
        if(len(variables) == 0):
            return []

      #  obsi = obsi.sel(plev=[3000,2000,1000,500,100])   
        obsi = obsi.sel(plev=[85000,70000,50000])#not in hpa, in pa
        data_np = obsi[variables].to_array().to_numpy()
        data_th = torch.from_numpy(data_np)
        return data_th
    
    def preprocess(self, prev_clim,clim,clim_next):
        out = dict()

        prev_surface_variables = self.xarr_to_tensor(clim, list(self.surface_variables))
        prev_plev_variables = self.xarr_to_tensor(clim,list(self.plev_variables))
       # prev_depth_variables = self.xarr_to_tensor(clim,list(self.depth_variables))

        input_surface_variables = self.xarr_to_tensor(clim, list(self.surface_variables))
        input_plev_variables = self.xarr_to_tensor(clim,list(self.plev_variables))
       # input_depth_variables = self.xarr_to_tensor(clim,list(self.depth_variables))

        target_surface_variables = self.xarr_to_tensor(clim_next, list(self.surface_variables))
        target_plev_variables = self.xarr_to_tensor(clim_next,list(self.plev_variables))
       # target_depth_variables = self.xarr_to_tensor(clim_next,list(self.depth_variables))

        

        prev_time = prev_clim.time.dt.strftime('%Y-%m').item()

        time = clim.time.dt.strftime('%Y-%m').item()
        next_time = clim_next.time.dt.strftime('%Y-%m').item()

        prev_month_index = int(prev_time.split('-')[-1]) - 1        
        cur_month_index = int(time.split('-')[-1]) - 1
        
        next_month_index = int(next_time.split('-')[-1]) - 1
    #    print(next_time)
   #     print(time)
       # prev_year_index = int(prev_time.split('-')[0]) - 1960
        cur_year_index = int(time.split('-')[0]) - 1960
       # next_year_index = int(next_time.split('-')[0]) - 1960
       # cur_year_forcings = np.broadcast_to(np.expand_dims(self.atmos_forcings[:,cur_year_index],(1,2)),(4,143,144)).astype(np.float32)
        cur_year_forcings = torch.Tensor(self.atmos_forcings[:,cur_year_index])
        cur_solar_forcings = torch.Tensor(self.solar_forcings[cur_year_index,cur_month_index])
       # cur_year_forcings = []
      #  cur_year_forcings = []
     #   print(input_plev_variables.shape,'dataloader')
        if(not self.generate_statistics):
            if(self.normalization == 'climatology'):
                
                prev_plev_variables = (prev_plev_variables - self.plev_means[cur_month_index]) / (self.plev_stds[cur_month_index])
                prev_surface_variables = (prev_surface_variables - self.surface_means[prev_month_index]) / (self.surface_stds[prev_month_index])
                input_surface_variables = (input_surface_variables - self.surface_means[cur_month_index]) / (self.surface_stds[cur_month_index])
           #     input_depth_variables = (input_depth_variables - self.depth_means[cur_month_index]) / (self.depth_stds[cur_month_index])
            #    print(target_surface_variables.shape)
                
                input_plev_variables = (input_plev_variables - self.plev_means[cur_month_index]) / (self.plev_stds[cur_month_index])
           #     print(next_month_index,'next_month_index')
                target_surface_variables = (target_surface_variables - self.surface_means[next_month_index]) / (self.surface_stds[next_month_index])
             #   target_depth_variables = (target_depth_variables - self.depth_means[next_month_index]) / (self.depth_stds[next_month_index])
                target_plev_variables = (target_plev_variables - self.plev_means[cur_month_index]) / (self.plev_stds[cur_month_index])
            #    print(target_surface_variables.shape)
            elif(self.normalization == 'normal' or self.normalization == 'spatial_normal'):
                input_surface_variables = (input_surface_variables - self.surface_means) / self.surface_stds
            #    input_depth_variables = (input_depth_variables - self.depth_means) / self.depth_stds

                target_surface_variables = (target_surface_variables - self.surface_means) / self.surface_stds
            #    target_depth_variables = (target_depth_variables - self.depth_means) / self.depth_stds
        
      

 
            if(self.flattened_plev and not self.generate_statistics):
                v,c,l,w = prev_plev_variables.shape
                prev_plev_variables = prev_plev_variables.reshape(v*c,l,w)
                prev_surface_variables = torch.concatenate([prev_surface_variables,prev_plev_variables],axis=0)
                input_plev_variables = input_plev_variables.reshape(v*c,l,w)
                input_surface_variables = torch.concatenate([input_surface_variables,input_plev_variables],axis=0)
                target_plev_variables = target_plev_variables.reshape(v*c,l,w)
                target_surface_variables = torch.concatenate([target_surface_variables,target_plev_variables],axis=0)

            if(self.z_normalize):
                prev_surface_variables = (prev_surface_variables - self.z_means) / self.z_stds
                input_surface_variables = (input_surface_variables - self.z_means) / self.z_stds
                target_surface_variables = (target_surface_variables - self.z_means) / self.z_stds

            if(self.delta):
           # surface_mask = (self.surface_delta_stds != 0)
           # depth_mask = (self.depth_delta_stds != 0)
               #target_surface_variables = (target_surface_variables[surface_mask] - input_surface_variables[surface_mask]) / (self.surface_delta_stds[surface_mask])
            #target_depth_variables = (target_depth_variables[depth_mask] - input_depth_variables[depth_mask]) / (self.depth_delta_stds[depth_mask])
               # target_plev_variables = (target_plev_variables - input_plev_variables) / self.plev_delta_stds
            ##    print(input_surface_variables.shape)
             #   print(target_surface_variables.shape)
                # print(self.surface_delta_stds.shape)
                # print(self.plev_delta_stds.shape)
                target_surface_variables = (target_surface_variables - input_surface_variables) / (torch.concatenate([self.surface_delta_stds,self.plev_delta_stds.reshape(8*3,1,1)]))
             #   print(self.surface_delta_stds.shape)
             #   target_depth_variables = (target_depth_variables - input_depth_variables) / self.depth_delta_stds
             #   print('after delta',target_surface_variables.shape)
                  #z normalize
          
            if(self.z_normalize):
                mask_val = self.mask_value
            else:
                mask_val = 0
            input_surface_variables = torch.nan_to_num(input_surface_variables,mask_val)
            input_plev_variables = torch.nan_to_num(input_plev_variables,mask_val)
           # input_depth_variables = torch.nan_to_num(input_depth_variables,0)
            target_surface_variables = torch.nan_to_num(target_surface_variables,mask_val)
            prev_surface_variables = torch.nan_to_num(prev_surface_variables,mask_val)
            target_plev_variables = torch.nan_to_num(target_plev_variables,mask_val)
            prev_plev_variables = torch.nan_to_num(prev_plev_variables,mask_val)
           # target_depth_variables = torch.nan_to_num(target_depth_variables,0)


        #REMOVE OUTLIERS
        # maximum = torch.quantile(input_surface_variables.reshape(34,-1),0.999,dim=1).to(batch['state_surface'].device)
        # minimum = torch.quantile(input_surface_variables.reshape(34,-1),0.001,dim=1).to(batch['state_surface'].device)
        # maximum = maximum.unsqueeze(1).unsqueeze(2).expand(-1,143,144).unsqueeze(0)
        # minimum = minimum.unsqueeze(1).unsqueeze(2).expand(-1,143,144).unsqueeze(0)
        # input_surface_variables = torch.clamp(input_surface_variables,min=minimum,max=maximum)

        
        # def remove_outliers(t:torch.Tensor):
            
        #     #REMOVE OUTLIERS
        #     maximum = torch.quantile(t.reshape(34,-1),0.99,dim=1)
        #     minimum = torch.quantile(t.reshape(34,-1),0.01,dim=1)
        #     maximum = maximum.unsqueeze(1).unsqueeze(2).expand(-1,143,144)
        #     minimum = minimum.unsqueeze(1).unsqueeze(2).expand(-1,143,144)
        #     t = torch.clamp(t,min=minimum,max=maximum)
        #     return t

        # input_surface_variables = remove_outliers(input_surface_variables)
        # prev_surface_variables = remove_outliers(prev_surface_variables)
        # target_surface_variables = remove_outliers(target_surface_variables)

        
        out.update(dict(
                    prev_state_surface=prev_surface_variables,
                    prev_state_level=prev_plev_variables,
                    state_surface=input_surface_variables,
                    state_level=input_plev_variables,
                #    state_depth=input_depth_variables.type(torch.float32),
                   # state_constant=self.land_mask.astype(np.float32),
                    state_constant=self.elevation_data,
                    next_state_surface=target_surface_variables,
                    next_state_level=target_plev_variables,
                #    next_state_depth=target_depth_variables.type(torch.float32),
                    time=time,
                    next_time=next_time,
                    forcings=cur_year_forcings,
                    solar_forcings=cur_solar_forcings))
        return out

    
    def denormalize(self, batch):
        device = batch['next_state_surface'].device
        cur_month = int(batch['time'][0].split('-')[-1]) - 1     
        next_month = int(batch['next_time'][0].split('-')[-1]) - 1    
        if(self.delta):
           # new_state_surface = ((batch['next_state_surface']*self.surface_delta_stds.to(device).unsqueeze(0)) + batch['state_surface'])
            new_state_surface=batch['next_state_surface'][:,:10].to(device)*self.surface_delta_stds.to(device).unsqueeze(0) + batch['state_surface'][:,:10].to(device)
            if(self.flattened_plev):
                new_state_plev=batch['next_state_surface'][:,10:].to(device)*self.plev_delta_stds.to(device).unsqueeze(0).reshape(1,8*3,1,1) + batch['state_surface'][:,10:].to(device)
                new_state_surface = torch.concatenate([new_state_surface,new_state_plev],axis=1)
        else:
            new_state_surface=batch['next_state_surface']
        if(self.normalization == 'climatology'):
            denorm_surface = lambda x,month_index: x[:,:10]*torch.from_numpy(self.surface_stds[month_index]).to(device) + torch.from_numpy(self.surface_means[month_index]).to(device)
            denorm_plev = lambda x,month_index: x[:,10:]*torch.from_numpy(self.plev_stds[month_index]).to(device).reshape(-1,8*3,143,144) + torch.from_numpy(self.plev_means[month_index]).to(device).reshape(-1,8*3,143,144)
           # denorm_surface = lambda x,month_index: x.to(device)*torch.from_numpy(self.surface_stds[month_index]).to(device) + torch.from_numpy(self.surface_means[month_index]).to(device)
        elif(self.normalization == 'normal' or self.normalization == 'spatial_normal'):
            denorm_surface = lambda x,month_index: x.squeeze().to(device)*torch.from_numpy(self.surface_stds).to(device) + torch.from_numpy(self.surface_means).to(device)
        batch = dict(
                    next_state_surface=torch.concatenate([denorm_surface(new_state_surface,next_month),denorm_plev(new_state_surface,next_month)],dim=1),
                    state_surface=torch.concatenate([denorm_surface(batch['state_surface'],cur_month),denorm_plev(batch['state_surface'],cur_month)],dim=1),
                   # time=batch['time']
        )
        return batch

    
    
    
    def __getitem__(self, i):
        file_id, prev_line_id = self.id2pt[i]
        line_id = prev_line_id + self.lead_time_months
        next_line_id = line_id + self.lead_time_months
        obs = xr.open_dataset(self.files[file_id], **self.xr_options)
        prev_clim = obs.isel(time=prev_line_id)
        clim = obs.isel(time=line_id)
        clim_next = obs.isel(time=next_line_id)
        
        out = self.preprocess(prev_clim,clim,clim_next)
        obs.close()
        return out
