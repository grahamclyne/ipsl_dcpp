#!/bin/bash
#SBATCH --job-name=sharpnesses     # job name
#SBATCH --ntasks-per-node=1          # this needs to correspond with # of GPUS
#SBATCH --cpus-per-task=8     # number of cores per tasks, see how many GPUs per node and take proportional amount of CPUs
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=00:20:00              # maximum execution time (HH:MM:SS)
#SBATCH --account=mlr@v100
#SBATCH --gpus=1

cd ${WORK}/ipsl_dcpp/ipsl_dcpp/
module load pytorch-gpu/py3/2.2.0
module load ffmpeg
srun python sharpness_comparison.py module=flow_no_climatology_no_delta
srun python sharpness_comparison.py module=flow_sub_pixel_no_delta
srun python sharpness_comparison.py module=flow_sub_pixel_elevation
srun python sharpness_comparison.py module=flow_sub_pixel
srun python sharpness_comparison.py module=flow_conv_head_layernorm
srun python sharpness_comparison.py module=ddpm_elevation_scaled_1000_timesteps
