from ipsl_dataset import IPSL_DCPP
import torch
train = IPSL_DCPP('train',generate_statistics=False,lead_time_months=1)
# train_dataloader = torch.utils.data.DataLoader(train,batch_size=2,shuffle=True,num_workers=1,)
from pangu import PanguWeather
model = PanguWeather(soil=True)
from hydra import compose, initialize
from omegaconf import OmegaConf


with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config")

train_dataloader = torch.utils.data.DataLoader(IPSL_DCPP('train',cfg.experiment.lead_time_months),batch_size=2,shuffle=False,num_workers=1)
batch = next(iter(train_dataloader))
out = model.forward(batch)