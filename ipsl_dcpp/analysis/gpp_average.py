#generate gpp averages of test data 

from ipsl_dataset import IPSL_DCPP
import torch
import hydra
from ipsl_dataset import surface_variables
from omegaconf import DictConfig



def get_decadal_ts_avgs(iter_ts,var_index):
    gpp_avgs = [[] for x in range(0,117)]
    count = 0
    run_count = 0
    while run_count < 10:
        data = next(iter_ts,None)
        print(data['time'],count)
        if(data is None):
            return gpp_avgs
        gpp_avgs[count % 117].append(data['state_surface'].squeeze()[var_index])
        if(count % 117 == 0 and count > 100):
            print('end ',count)
            count = 0
            run_count = run_count + 1
        else:
            count = count + 1
    return gpp_avgs




@hydra.main(version_base=None,config_path='./conf',config_name="config.yaml")
def main(cfg: DictConfig):
    test = IPSL_DCPP('test',lead_time_months=1)
    test_dataloader = torch.utils.data.DataLoader(test,batch_size=1,shuffle=False,num_workers=8)
    iter_ts = iter(test_dataloader)
    surface_var_name = 'gpp'
    var_index = surface_variables.index(surface_var_name)
    gpp_avgs = get_decadal_ts_avgs(iter_ts,var_index)
    import pickle
    output = open('gpp_avgs.pkl', 'wb')

    # Pickle dictionary using protocol 0.
    pickle.dump(gpp_avgs, output)
    #np.save('gpp_averages',np.stack(gpp_avgs))    

    
    
if __name__ == '__main__':
    main()