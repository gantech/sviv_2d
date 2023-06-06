#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --account=sviv
#SBATCH --job-name=af_smd
#SBATCH --output=out.%x_%j
#SBATCH --array=1-3

export SPACK_MANAGER=~/spack-manager
source ${SPACK_MANAGER}/start.sh
spack-start
spack env activate -d ~/spack-manager/environments/sviv_smd_run/ 
spack load nalu-wind

# Set directory for the test case
line_str=$(printf '%dp' ${SLURM_ARRAY_TASK_ID})
directory=$(sed -n $line_str list_of_cases)

cd $directory

srun -n 4 naluX -i *.yaml &> log 
