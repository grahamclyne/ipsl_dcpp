#!/bin/bash
#SBATCH --job-name=diffusion_rollout     # job name
#SBATCH --ntasks-per-node=2          # this needs to correspond with # of GPUS
#SBATCH --cpus-per-task=10     # number of cores per tasks, see how many GPUs per node and take proportional amount of CPUs
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --time=05:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --account=mlr@a100
#SBATCH --gpus=2
cd ${WORK}/ipsl_dcpp/ipsl_dcpp/
# source ../../miniconda3/etc/profile.d/conda.sh
# conda init bash
# conda activate env_gpu5 
module load cpuarch/amd

module load pytorch-gpu/py3/2.2.0
module load ffmpeg
#module load pytorch-cpu/py3/1.7.1 
srun python diffusion_evaluation.py cluster=jean_zay_a100
