surface_variables = ['baresoilFrac',
 'c3PftFrac',
 'c4PftFrac',
 'cLeaf',
 'cLitter',
 'cLitterAbove',
 'cLitterBelow',
 'cProduct',
 'cRoot',
 'cSoilFast',
 'cSoilMedium',
 'cSoilSlow',
 'cVeg',
 'cropFrac',
 'mrsos',
 'evspsblsoi',
 'evspsblveg',
 'fHarvest',
 'fLitterSoil',
 'fVegLitter',
 'grassFrac',
 'lai',
 'mrfso',
 'mrro',
 'mrros',
 'mrso',
 'prveg',
 'rGrowth',
 'rMaint',
 'ra',
# 'residualFrac',
 'rh',
 'tran',
 'treeFrac',
 'treeFracPrimDec',
 'treeFracPrimEver',
 'ci',
 'clivi',
 'clt',
 'clwvi',
 'evspsbl',
 'hfls',
 'hfss',
 'hurs',
 'huss',
 'pr',
 'prc',
 'prsn',
 'prw',
 'ps',
 'psl',
 'rlds',
 'rldscs',
 'rlus',
 'rlut',
 'rlutcs',
 'rsds',
 'rsdscs',
 'rsdt',
 'rsus',
 'rsuscs',
 'rsut',
 'rsutcs',
 'rtmt',
 'sfcWind',
 'tas',
 'tasmax',
 'tasmin',
 'tauu',
 'tauv',
 'ts',
 'uas',
 'vas',
 'cLand',
 'cLitterGrass',
 'cLitterSubSurf',
 'cLitterSurf',
 'cLitterTree',
 'cMisc',
 'cOther',
 'cSoil',
 'cSoilGrass',
 'cSoilTree',
 'cStem',
 'cVegGrass',
 'cVegTree',
 'cWood',
 'cropFracC3',
 'cropFracC4',
 'evspsblpot',
 'fAnthDisturb',
 'fDeforestToAtmos',
 'fDeforestToProduct',
 'fHarvestToAtmos',
 'fHarvestToProduct',
 'fLuc',
 'fProductDecomp',
 'grassFracC3',
 'grassFracC4',
 'intuadse',
 'intuaw',
 'intvadse',
 'intvaw',
 'mrlso',
 'mrtws',
 'prhmax',
 'raGrass',
 'raTree',
 'rhGrass',
 'rhLitter',
 'rhSoil',
 'rhTree',
 'rls',
 'rss',
 'sconcdust',
 'sconcso4',
 'sconcss',
 'sfcWindmax',
 'treeFracBdlDcd',
 'treeFracBdlEvg',
 'treeFracNdlDcd',
 'treeFracNdlEvg',
 'vegFrac',
    'npp',
                   'nbp',
                   'gpp',
                   'nppGrass',
                   'nppOther',
                   'nppStem',
                   'nppTree',
                   'nep',
                   'nppLeaf',
                   'nppRoot',
                   'nppWood',
                   'gppGrass',
                   'gppTree'
                  ]

plev_variables = ['hur','hus','o3','ta','ua','va','wap','zg']
landtype_variables = ['landCoverFrac']
depth_variables = ['mrsfl','mrsol','mrsll']
import glob
import os 
from tqdm import tqdm
import xarray as xr
import torch
import numpy as np
import pickle
work = os.environ['WORK']
scratch = os.environ['SCRATCH']
class IPSL_DCPP(torch.utils.data.Dataset):
    def __init__(self,domain,generate_statistics=False):
        self.files = list(glob.glob(f'{scratch}/*.nc'))
        self.files = dict(
                    all_=[str(x) for x in self.files],
                    train=[str(x) for x in self.files if any(substring in x for substring in [str(x) for x in list(range(1960,2000))])],
                    val = [str(x) for x in self.files if any(substring in x for substring in [str(x) for x in list(range(2011,2014))])],
                    test = [str(x) for x in self.files if any(substring in x for substring in [str(x) for x in list(range(2014,2017))])])[domain]
        self.nfiles = len(self.files)
        self.xr_options = dict(engine='netcdf4', cache=True)
        self.lead_time_months = 6
      #  self.input_means = np.expand_dims(np.load(f'{work}/ipsl_dcpp/data/input_means.npy'),axis=(1,2))
      #  self.input_stds = np.expand_dims(np.load(f'{work}/ipsl_dcpp/data/input_stds.npy'),axis=(1,2))
      #  self.target_means = np.expand_dims(np.load(f'{work}/ipsl_dcpp/data/target_means.npy'),axis=(1,2))
      #  self.target_stds = np.expand_dims(np.load(f'{work}/ipsl_dcpp/data/target_stds.npy'),axis=(1,2))
        self.generate_statistics=generate_statistics
        self.timestamps = []
        self.id2pt_path = f'{work}/ipsl_dcpp/data/id2pt.pkl'
        if os.path.exists(self.id2pt_path):
            with open(self.id2pt_path, 'rb') as handle:
                self.id2pt = pickle.load(handle)
        else:
            for fid, f in tqdm(enumerate(self.files)):
                with xr.open_dataset(f, **self.xr_options) as obs:
                    var_id = f.split('.')[0][-1]
                    file_stamps = [(fid, i, t,var_id) for (i, t) in enumerate(obs.time.to_numpy())]
                    #if doing autoregressive - don't include the last -leadtime- amount of each timeseries to avoid indexing overflow issues
                    self.timestamps.extend(file_stamps[:-self.lead_time_months+1])
                    #self.timestamps.extend(file_stamps)
            self.timestamps = sorted(self.timestamps, key=lambda x: x[-1]) # sort by timestamp
            self.id2pt = {i:(file_id, line_id) for (i, (file_id, line_id, var_id,s)) in enumerate(self.timestamps)}
            with open(self.id2pt_path, 'wb') as handle:
                pickle.dump(self.id2pt,handle)
                
    def __len__(self):
        return len(self.id2pt)
    
    def xarr_to_tensor(self, obsi,variables):
        data_np = obsi[variables].to_array().to_numpy()
        data_th = torch.from_numpy(data_np)      
        return data_th
    
    def preprocess(self, clim,clim_next):
        out = dict()
        input_surface_variables = self.xarr_to_tensor(clim, surface_variables)
        input_plev_variables = self.xarr_to_tensor(clim,plev_variables)
        input_depth_variables = self.xarr_to_tensor(clim,depth_variables)
        
        
        target_surface_variables = self.xarr_to_tensor(clim_next, surface_variables)
        target_plev_variables = self.xarr_to_tensor(clim_next,plev_variables)
        target_depth_variables = self.xarr_to_tensor(clim_next,depth_variables)
        time = clim.time.dt.strftime('%Y-%m').item()
        next_time = clim_next.time.dt.strftime('%Y-%m').item()

        if(not self.generate_statistics):
            input_surface_variables = (input_surface_variables - self.input_means) / self.input_stds
            input_plev_variables = (input_plev_variables - self.target_means) / self.target_stds
            input_depth_variables = (input_depth_variables - self.input_means) / self.input_stds
            target_surface_variables = (target_surface_variables - self.target_means) / self.target_stds
            target_plev_variables = (target_plev_variables - self.input_means) / self.input_stds
            target_depth_variables = (target_depth_variables - self.target_means) / self.target_stds
            
            input_surface_variables = input_surface_variables.nan_to_num(inputs,0)
            input_plev_variables = input_plev_variables.nan_to_num(targets,0)
            input_depth_variables = input_depth_variables.nan_to_num(inputs,0)
            target_surface_variables = target_surface_variables.nan_to_num(targets,0)
            target_plev_variables = target_plev_variables.nan_to_num(inputs,0)
            target_depth_variables = target_depth_variables.nan_to_num(targets,0)
            
        out.update(dict(
                    surface_inputs=input_surface_variables,
                    plev_inputs=input_plev_variables,
                    depth_inputs=input_depth_variables,
                    surface_outputs=target_surface_variables,
                    plev_outputs=target_plev_variables,
                    depth_targets=target_depth_variables,
                    time=time,
                    next_time=next_time
                ))
        return out

    def __getitem__(self, i):
        file_id, line_id = self.id2pt[i]
        next_line_id = line_id + self.lead_time_months
        obs = xr.open_dataset(self.files[file_id], **self.xr_options)
        clim = obs.isel(time=line_id)
        clim_next = obs.isel(time=next_line_id)
        out = self.preprocess(clim,clim_next)
        obs.close()
        return out
