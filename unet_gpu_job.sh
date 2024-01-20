#!/bin/bash
#SBATCH --job-name=unet_ipsl     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=2          # number of MPI tasks per node, this needs to correspond with # of GPUS
#SBATCH --gres=gpu:2                 # number of GPUs per node
#SBATCH --cpus-per-task=2            # number of cores per tasks, see how many GPUs per node and take proportional amount of CPUs
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=00:10:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=pytorch_stl10%j.out # output file name
#SBATCH --error=pytorch_stl10%j.err  # error file name
##SBATCH --partition=gpu_p2 
#SBATCH --account=mlr@v100
##SBATCH -C a100
#to use a100's need to update pytorch

cd ${WORK}
export WANDB_MODE=offline
export WANDB_API_KEY=c1f678c655920120ec68e1dc542a9f5bab02dbfa

source miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate env_gpu3 

srun python ./ipsl_dcpp/train.py
