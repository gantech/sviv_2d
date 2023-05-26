# SVIV Simulations in 2D


## Model Construction

### OpenFast

### Python Scripts

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


## Nalu Wind Runs


