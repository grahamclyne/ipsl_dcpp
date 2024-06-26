#!/bin/bash
#SBATCH --job-name=gpp_test_average    # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # this needs to correspond with # of GPUS
#SBATCH --cpus-per-task=8            # number of cores per tasks, see how many GPUs per node and take proportional amount of CPUs
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=00:05:00              # maximum execution time (HH:MM:SS)
#SBATCH --account=mlr@cpu

cd ${WORK}/ipsl_dcpp/

source ../miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate env_gpu5 

srun python gpp_average.py experiment='one_month_full_variable_a100'
