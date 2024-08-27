#from ipsl_dcpp.model.ipsl_dataset import IPSL_DCPP
import torch
import lightning as pl
#from ipsl_dcpp.model.pangu import PanguWeather
from hydra import compose, initialize
from omegaconf import OmegaConf
import numpy as np
import hydra
import os
import pickle
import io
import numpy as np
from matplotlib import animation
import xarray as xr 
import matplotlib.pyplot as plt 
#os.environ['SLURM_NTASKS_PER_NODE'] = '1'
#torch.set_default_dtype(torch.float32)
# os.environ["CUDA_VISIBLE_DEVICES"]=""
#torch.set_default_tensor_type(torch.FloatTensor)

with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config")
pl.seed_everything(cfg.seed)
train = hydra.utils.instantiate(
    cfg.dataloader.dataset,domain='train',debug=False
)
train_loader = torch.utils.data.DataLoader(train, 
                                            batch_size=1,
                                            num_workers=0,
                                            shuffle=True) 
ts = train.timestamps.copy()

import datetime
def inc_time(batch_time):
    batch_time = datetime.datetime.strptime(batch_time,'%Y-%m')
    if(batch_time.month == 12):
        year = batch_time.year + 1
        month = 1
    else:
        year = batch_time.year
        month = batch_time.month + 1
    if(month < 10):
        return f'{year}-0{month}'
    else:
        return f'{year}-{month}'

ts.sort(key=lambda tup: tup[2])
def search_by_value(d, search_value):
    found_keys = []
    for key,value in d.items():
        if value == search_value:
            return key

num_ensembles = 7
out_mapping = {}
#want to make a map of each year to x num variables
time = '1961-01'
means = []
stds = []
for _ in range(0, 500):
    print(time)
    vals = []
    indices = [(x[0],x[1]) for x in list(filter(lambda x: time in str(x[2]), ts))]
    time=inc_time(time)
    for index in indices:
        vals.append(train.__getitem__(search_by_value(train.id2pt,index))['state_surface'][0])
    output = torch.stack(vals)
    print(output.shape)
    mean = output.mean()
    std = output.std()
    means.append(mean)
    stds.append(std)
   # out_mapping[str(ts[time_index*(num_ensembles+1)][2])[:7]] =  torch.stack(vals)
    print(mean,std)
   # print(str(ts[time_index*(num_ensembles+1)][2])[:7])
import pickle
mean_file = open('long_means', 'ab')
mean_std = open('long_stds', 'ab')

# source, destination
pickle.dump(means, mean_file) 
pickle.dump(stds, mean_std) 