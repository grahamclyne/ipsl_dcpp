input_variables = ['baresoilFrac',
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
 'gpp',
 'grassFrac',
 'lai',
 'mrfso',
 'mrro',
 'mrros',
 'mrso',
 'nbp',
 'npp',
 'nppLeaf',
 'nppRoot',
 'nppWood',
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
 'gppGrass',
 'gppTree',
 'grassFracC3',
 'grassFracC4',
 'intuadse',
 'intuaw',
 'intvadse',
 'intvaw',
 'mrlso',
 'mrtws',
 'nep',
 'nppGrass',
 'nppOther',
 'nppStem',
 'nppTree',
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
 'vegFrac']

# target_variables=['npp','nbp','gpp','nppGrass','nppOther','nppStem','nppTree','nep','nppLeaf','nppRoot','nppWood',]


import glob
import os 
from tqdm import tqdm
import xarray as xr
import torch
import numpy as np
work = os.environ['WORK']
scratch = os.environ['SCRATCH']
class IPSL_DCPP(torch.utils.data.Dataset):
    def __init__(self,domain):
        self.files = list(glob.glob(f'{scratch}/*.nc'))
        self.files = dict(
                    all_=[str(x) for x in self.files],
                    train=[str(x) for x in self.files if not '1960_8.nc' in x],
                    val = [str(x) for x in self.files if '1960_8.nc' in x],
                    test = [str(x) for x in self.files if '2020' in x])[domain]
        self.nfiles = len(self.files)
        self.xr_options = dict(engine='netcdf4', cache=True)

        self.timestamps = []
        self.lead_time_months = 6
        self.means = np.expand_dims(np.load(f'{work}/ipsl_dcpp/data/means.npy'),axis=(1,2))

        self.stds = np.expand_dims(np.load(f'{work}/ipsl_dcpp/data/std_devs.npy'),axis=(1,2))

        for fid, f in tqdm(enumerate(self.files)):
            with xr.open_dataset(f, **self.xr_options) as obs:
                var_id = f.split('.')[0][-1]
                file_stamps = [(fid, i, t,var_id) for (i, t) in enumerate(obs.time.to_numpy())]
                #don't include the last -leadtime- amount of each timeseries to avoid indexing overflow issues
                self.timestamps.extend(file_stamps[:-self.lead_time_months])


        self.timestamps = sorted(self.timestamps, key=lambda x: x[-1]) # sort by timestamp
        self.id2pt = {i:(file_id, line_id) for (i, (file_id, line_id, var_id,s)) in enumerate(self.timestamps)}
    def __len__(self):
        return len(self.id2pt)
    
    def xarr_to_tensor(self, obsi,variables):
        
        data_np = obsi[variables].to_array().to_numpy()
        data_th = torch.from_numpy(data_np)      
        return data_th
    
    def preprocess(self, clim,clim_next):
        out = dict()
        inputs = self.xarr_to_tensor(clim, input_variables)
        targets = self.xarr_to_tensor(clim_next, input_variables)
        time = clim.time.dt.strftime('%Y-%m').item()
        next_time = clim_next.time.dt.strftime('%Y-%m').item()
        inputs = (inputs - self.means) / self.stds
        targets = (targets - self.means) / self.stds
        inputs = torch.nan_to_num(inputs,0)
        targets = torch.nan_to_num(inputs,0)
        out.update(dict(
                    inputs=inputs, 
                    targets=targets,
                    time=time,
        next_time=next_time))
        return out

    def __getitem__(self, i):
        file_id, line_id = self.id2pt[i]
        next_line_id = line_id + self.lead_time_months
        obs = xr.open_dataset(self.files[file_id], **self.xr_options)
        clim = obs.isel(time=line_id)
        clim_next = obs.isel(time=next_line_id)
        out = self.preprocess(clim,clim_next)
      #  if out['inputs'].isnan().any().item():
      #   #   print(i, file_id, line_id, self.files[file_id])
        #    print('NaN values detected !')  
        obs.close()
        return out
