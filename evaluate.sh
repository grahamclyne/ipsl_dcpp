#!/bin/bash
#SBATCH --job-name=ipsl_evaluate     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # this needs to correspond with # of GPUS
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=3            # number of cores per tasks, see how many GPUs per node and take proportional amount of CPUs
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=01:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=eval_%j.out # output file name
#SBATCH --error=eval_%j.err  # error file name
#SBATCH --account=mlr@v100


cd ${WORK}
source miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate env_gpu5 

srun python ./ipsl_dcpp/evaluate.py
