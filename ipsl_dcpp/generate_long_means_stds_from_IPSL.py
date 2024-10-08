import torch
import lightning as pl
from hydra import compose, initialize
import hydra
import datetime

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
                                            shuffle=False) 
ts = train.timestamps.copy()

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
    for key,value in d.items():
        if value == search_value:
            return key

out_mapping = {}
#want to make a map of each year to x num variables
time = '1961-02'
means = []
stds = []
outputs = []
var_index = 8
for _ in range(0, 590):
    print(time)
    vals = []
    indices = [(x[0],x[1]) for x in list(filter(lambda x: time in str(x[2]), ts))]
    time=inc_time(time)
    for index in indices:
        batch = train.__getitem__(search_by_value(train.id2pt,index))
        batch = {k:[batch[k]] if k == 'time' or k == 'next_time' else batch[k].unsqueeze(0) for k in batch.keys()}  #simulate lightnings batching dimension
        # denormed = train.denormalize(batch)
        vals.append(batch['state_surface'][0,var_index]) #only get tas
    output = torch.stack(vals)
    # print(output.shape)
    # print(output.mean(axis=0).shape)
    outputs.append(output.mean(axis=0))
    # mean = output.mul(train.lat_coeffs_equi[0][0]).nanmean(axis=(-1,-2))

        # print(batch.shape)
    
    # mean = output.mean()
    # std = output.std()
    # means.append(mean.mean())
    # stds.append(mean.std())
    #outputs.append(output)
   # out_mapping[str(ts[time_index*(num_ensembles+1)][2])[:7]] =  torch.stack(vals)
   # print(mean,std)
   # print(str(ts[time_index*(num_ensembles+1)][2])[:7])
#mean_file = open(f'{var_index}_normal_normalized_long_means_ipsl.pt','wb')
#mean_std = open(f'{var_index}_normal_normalized_long_stds_ipsl.pt','wb')
output_file = open(f'{var_index}_spatial_ensemble_batch_mean.pt','wb')

# source, destination
#torch.save(torch.stack(means), mean_file) 
#torch.save(torch.stack(stds), mean_std) 
torch.save(torch.stack(outputs), output_file) 