#!/bin/bash
#SBATCH --job-name=reproj_omon     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --cpus-per-task=4      # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=02:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --account=mlr@cpu
cd ${WORK}/ipsl_dcpp/ipsl_dcpp

#source ../miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate esmf_test1


srun python ./utils/reproject_omon.py