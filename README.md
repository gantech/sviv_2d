# SVIV Simulations in 2D


## Model Construction

### OpenFast

The main outputs needed from OpenFast for model construction are the Mass and Stiffness matrices from a dynamic BeamDyn analysis. 
```
cd OpenFAST/BeamDyn/dynamic
beamdyn_driver bd_driver.inp 
```
The results file "bd_driver.BD.sum.yaml" is used to load the mass and stiffness matrices. Copy this file to the top level python directory. 

### Python Scripts

Python scripts use the following packages:
- numpy
- scipy
- pyyaml
- netCDF4

BeamDyn summary file from the dynamic simulation (with no loads or displacements) should be put at the top level python directory to be loaded by several scripts.


#### python/ConstructModel

Run the script "python/ConstructModel/chord_prop_model.py" to generate a yaml file describing a 3 DOF model for the aerodynamic loads being proportional to the chord.

This script generates the yaml file "chord_3dof.yaml" with the mass, stiffness, damping, and force transformation matrices.

#### python/DistributedLoad

Used for verifying extracted properties against BeamDyn simulations.

#### python/DetermineRunSets

Script for generating a list of velocities of interest. Script also creates a plot to visualize what coverage of the frequency response for a linear system with constant external force would be covered.

#### python/ModalAnalysis

The script "load_M_K.py" does an eigenvalue analysis of the BeamDyn matrices and prints frequencies. This can be used as a sanity check. This script requires copying a yaml summary file from a dynamic BeamDyn simulation to be able to extract the mass and stiffness matrices.


#### python/HHTAlpha

Misc. scripts for checking HHT Alpha-method implementation for time integration. 

"hht_test.py" is used for verification of the nalu-wind simulation against known lift/drag values. This file requires the .yaml file produced by constructing a model. It currently uses "chord_3dof.yaml".



### Verifying FEM Functionality

1. From the root directory of the repo, run distributed load cases with BeamDyn:
```
cd ./OpenFAST/BeamDyn/fullLengthStatic
source ./run_dist.sh
```
This script uses some environmental variables to load spack and BeamDyn. 

2. Navigate to the python directory and copy the outputs (starting from the root directory of the repo again):
```
cd python/DistributedLoad
source ./copy_results.sh
```

3. Run the python comparison:
```
python verify_dist_loads.py
```

#### Rotation Values

1. Navigate to and run the BeamDyn files for the rotation cases:
```
cd OpenFAST/BeamDyn/rotation_center
source axis_runs.sh
```

2. Navigate to the python script folder and copy the BeamDyn outputs:
```
cd ../../../python/RotationChecks/
source copy_bd_outs.sh
```

3. Run the python script "python/RotationChecks/rot_compare.py". This script indicates that displaying the top section and applying an identical moment at the half span results in translation of the tip. Therefore, should not need to add any translation due to rotation about an offset axis to the Nalu wind simulations. Some error in the comparisons here, likely because moving the top cross section changes the lower stiffness properties due to the global shape functions. 

## Nalu Wind Runs

To generate inputs:
```
cd nalu-wind
python utilities/gen_naluwind_inp_files.py
cd nalu_runs
mkdir job_list
cd job_list
find .. -name freq_* | sort -n > list_of_cases
sbatch job_submit.slurm
```

Note that in job_submit.slurm, the array of inputs is indexed from 1 to the number of lines in the file job_list.

