import xarray as xr
import os
store_dir = os.environ['STORE'] 
year = 1960
variation = 1
Amon =  xr.open_mfdataset(f'{store_dir}/s{year}-r{variation}i1p1f1/Amon/*.nc',compat='minimal')
#get climatology over train period for a year period
train = IPSL_DCPP('train',generate_statistics=True,lead_time_months=1)
train_dataloader = torch.utils.data.DataLoader(train,batch_size=1,shuffle=False)
out = []
iter_batch = iter(train_dataloader)
for count in range(10):
    print(count)
    year = []
    for _ in range(12):
        batch = next(iter_batch)
        surface = batch['state_surface'].squeeze().nanmean((-2,-1))
        year.append(surface)
        
    
    out.append(np.stack(year))
    
fin_out = np.stack(out)
np.save('climatology_from_train.npy',fin_out.mean(axis=0))