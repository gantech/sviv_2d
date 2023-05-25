# SVIV Simulations in 2D


## Model Construction

### OpenFast

### Python Scripts

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


