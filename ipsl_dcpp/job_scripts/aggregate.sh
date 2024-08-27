#!/bin/bash
#SBATCH --job-name=aggregate     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --cpus-per-task=1      # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=03:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --account=mlr@cpu
cd ${WORK}/ipsl_dcpp/ipsl_dcpp

module load pytorch-gpu/py3/2.2.0

srun python ./utils/aggregate_xarrays.py
