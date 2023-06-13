#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --account=sviv
#SBATCH --job-name=ramp_demo
#SBATCH --output=out.%x_%j

export SPACK_MANAGER=~/spack-manager
source ${SPACK_MANAGER}/start.sh
spack-start
spack env activate -d ~/spack-manager/environments/sviv_smd_debug/ 
spack load nalu-wind

srun -n 2 naluX -i airfoil_osc.yaml &> log
