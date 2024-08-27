#!/bin/bash
#SBATCH --job-name=diffusion_evaluation     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # this needs to correspond with # of GPUS
#SBATCH --cpus-per-task=8           # number of cores per tasks, see how many GPUs per node and take proportional amount of CPUs
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=05:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --account=mlr@v100
#SBATCH --gpus=1


cd ${WORK}/ipsl_dcpp/ipsl_dcpp/
module load pytorch-gpu/py3/2.2.0

srun python diffusion_evaluation.py 
