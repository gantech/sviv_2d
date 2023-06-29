#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --account=sviv
#SBATCH --job-name=prescribed
#SBATCH --output=out.%x_%j
#SBATCH --array=1-7

export SPACK_MANAGER=~/spack-manager
source ${SPACK_MANAGER}/start.sh
spack-start
spack env activate -d ~/spack-manager/environments/sviv_prescribe/ 
spack load nalu-wind

# Set directory for the test case
caselist=$(printf "list_of_cases_%02d" ${SLURM_ARRAY_TASK_ID})

for i in `cat ${caselist}`
do
    cd $i
    srun -n 10 naluX -i *.yaml &> log 
    cd -
done

wait

