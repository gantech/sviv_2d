# SVIV Simulations in 2D


## Model Construction

### OpenFast

### Python Scripts

Python scripts use the following packages:
- numpy
- scipy
- pyyaml


#### python/ConstructModel

Run the script "python/ConstructModel/chord_prop_model.py" to generate a yaml file describing a 3 DOF model for the aerodynamic loads being proportional to the chord.

This script generates the yaml file "chord_3dof.yaml" with the mass, stiffness, and damping matrices.

#### python/DistributedLoad

Used for verifying extracted properties against BeamDyn simulations.

#### python/DetermineRunSets

Script for generating a list of velocities of interest. Script also creates a plot to visualize what coverage of the frequency response for a linear system with constant external force would be covered.

#### python/ModalAnalysis

The script "load_M_K.py" does an eigenvalue analysis of the BeamDyn matrices and prints frequencies. This can be used as a sanity check. This script requires copying a yaml summary file from a dynamic BeamDyn simulation to be able to extract the mass and stiffness matrices.


#### python/HHTAlpha

Misc. scripts for checking HHT Alpha-method implementation for time integration. These do not serve any important purpose other than scratch work. 



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


