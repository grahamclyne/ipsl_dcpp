#submitit file
import submitit
import hydra
from omegaconf import DictConfig

from omegaconf import OmegaConf

try:
    OmegaConf.register_new_resolver("eval", eval)
except:
    pass

from train import main as train_main

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    aex = submitit.AutoExecutor(folder=cfg.cluster.folder, cluster='slurm')
    aex.update_parameters(**cfg.cluster.launcher) # original launcher
    aex.submit(train_main, cfg)

if __name__ == '__main__':
    main()
