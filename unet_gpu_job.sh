#!/bin/bash
#SBATCH --job-name=unet_ipsl     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=3            # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=00:15:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=pytorch_stl10%j.out # output file name
#SBATCH --error=pytorch_stl10%j.err  # error file name
#SBATCH --account=mlr@v100
cd ${WORK}
export WANDB_MODE=offline
export WANDB_API_KEY=c1f678c655920120ec68e1dc542a9f5bab02dbfa

source miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate env_gpu3 

srun python ./ipsl_dcpp/train.py