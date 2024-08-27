#!/bin/bash
#SBATCH --job-name=diffusion_rollout     # job name
#SBATCH --ntasks-per-node=1          # this needs to correspond with # of GPUS
#SBATCH --cpus-per-task=10      # number of cores per tasks, see how many GPUs per node and take proportional amount of CPUs
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=10:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --account=mlr@v100
#SBATCH --gpus=1

cd ${WORK}/ipsl_dcpp/ipsl_dcpp/
# source ../../miniconda3/etc/profile.d/conda.sh
# conda init bash
# conda activate env_gpu5 
module load pytorch-gpu/py3/2.2.0
module load ffmpeg
#module load pytorch-cpu/py3/1.7.1 
srun python long_rollout.py
