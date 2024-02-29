#!/bin/bash
#SBATCH --job-name=unet_ipsl     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --cpus-per-task=1            # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=aggregate_%j.out # output file name
#SBATCH --error=aggregate_%j.err  # error file name
#SBATCH --account=mlr@cpu
cd ${WORK}

source miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate env_gpu3 

srun python ./ipsl_dcpp/aggregate_xarrays.py
