#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --account=sviv
#SBATCH --job-name=smd_free_vib
#SBATCH --output=out.%x_%j

export SPACK_MANAGER=~/spack-manager
source ${SPACK_MANAGER}/start.sh
spack-start
spack env activate -d ~/spack-manager/environments/sviv_smd_debug/ 
spack load nalu-wind

srun -n 4 naluX -i airfoil_osc.yaml &> log
