from lightning.pytorch.loggers import WandbLogger
import os
from hydra import compose, initialize
from omegaconf import OmegaConf
import omegaconf
import sys
import subprocess
import wandb
api = wandb.Api()

#with initialize(version_base=None, config_path="conf"):
#    cfg = compose(config_name="config",overrides=["experiment=one_month_single_variable_v100"])

run_id = sys.argv[1]
subprocess.run(f"wandb sync wandb/offline-run*-{run_id}", shell=True)
#out = subprocess.run('readlink -f wandb/latest-run',shell=True,capture_output=True,text=True)
#run_name = out.stdout.strip('\n').split('-')[-1]
#run_id = cfg.run_id
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_API_KEY'] = 'c1f678c655920120ec68e1dc542a9f5bab02dbfa'

#conf = OmegaConf.to_container(
#    cfg.experiment, resolve=True, throw_on_missing=True
#)


run = api.run(f"gclyne/ipsl_dcpp_emulation/{run_id}")
#for key,value in conf.items():
#    print(key,value)
#    run.config[key] = value
run.name = run_id
run.upload_file(f'images/{run_id}_one_prediction.png')
run.upload_file(f'images/{run_id}_rollout_means.png')
run.upload_file(f'images/{run_id}_seasonal_predictions_with_errors.png')

run.update()