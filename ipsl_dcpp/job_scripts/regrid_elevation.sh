#!/bin/bash
#SBATCH --job-name=regrid_elevation     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --cpus-per-task=8     # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=01:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --account=mlr@cpu
cd ${WORK}/ipsl_dcpp/ipsl_dcpp

#source ../../miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate esmf_temp1


srun python ./regrid_elevation.py