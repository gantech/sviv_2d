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

Run the script "python/ConstructModel/create_model_funs.py" to generate a yaml file describing a 3 DOF model for the aerodynamic loads being proportional to the chord.
This file also contains functions that can be called to construct models for the IEA 15MW turbine (construct_IEA15MW_chord) and a more general function (construct_3dof). 

When considering different angles of attack, new 3 DOF model should be constructed to give mass and stiffness properties along the coordinate system of the CFD simulations (use construct_IEA15MW_chord and vary the angle of attack). 
If the CFD simulation coordinate system is fixed, then new models do not need to be constructed. 

This script generates the yaml file "chord_3dof.yaml" with the mass, stiffness, damping, and force transformation matrices. Documentation for the interpretation and use of these matrices should be found in the paper. 
The 3 DOF matrices can be loaded from this file into python as
```
# Load YAML file
with open(yamlfile) as f:
    data = list(yaml.load_all(f, Loader=SafeLoader))

dict_data = data[-1]

# Extract Matrices 
M3dof = np.array(dict_data['mass_matrix']).reshape(3,3)
C3dof = np.array(dict_data['damping_matrix']).reshape(3,3)
K3dof = np.array(dict_data['stiffness_matrix']).reshape(3,3)
T3dof = np.array(dict_data['force_transform_matrix']).reshape(3,3)
```

#### python/HHTAlpha

Misc. scripts for checking HHT Alpha-method implementation for time integration. 

"hht_test.py" can be used for verification of the nalu-wind simulation against known lift/drag values. This file requires the .yaml file produced by constructing a model. It currently uses "chord_3dof.yaml". This file also provides an example time integration. 

The script "unit_tests.py" was used to produce expected results for the Nalu-Wind unit tests.


#### python/Visualization

In the "single_results" folder, one can download a *.nc file with data from a Nalu-Wind simulation. Then the script "vis_sim.py" can be run to quickly plot time histories for displacements and forces (in both physical and modal domains). A yaml file for the 3DOF model is also needed in this folder to get the force transformation matrix so that the forces can be interpretted with the 3 DOF model. Note that if multiple angles of attack are considered, then the 3 DOF model needs to change appropriately here. Alternatively, the nc files from Nalu-Wind could be updated to include the force transformation matrix and the "vis_sim.py" file updated appropriately.  

#### python/ModalAnalysis

The script "load_M_K.py" does an eigenvalue analysis of the BeamDyn matrices and prints frequencies. This can be used as a sanity check. This script requires copying a yaml summary file from a dynamic BeamDyn simulation to be able to extract the mass and stiffness matrices. This script does not do any other useful cacluations. 

The script "mode_shape_plot.py" produces figures to visualize the mode shapes and the load distribution. These are useful for the presentation/paper. 

The script "static_modal_load.py" produces a figure comparing the displacement for a static load cases against projecting onto the first mode. Since the first mode is generally expected to be a good approximation of this bending behavior, and this is confirmed by the plots. This is a sanity check, that is not important for other details.


#### python/PFF

This folder is used for the peak finding and fitting method for system identification. By downloading an .nc file from a Nalu-Wind simulation, the script "peak_filter_fit.py" can be executed and produce a set of plots. 
This algorithm can perform better than Hilbert Transforms for extracting damping properties from transient data. 

The reference for this algorithm is:
M. Jin, W. Chen, M.R.W. Brake, H. Song,
Identification of Instantaneous Frequency and Damping From Transient Decay Data,
Journal of Vibration and Acoustics, 2020.



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

#### python/SimpleBeam

This directory contains information for verifying the 3DOF construction against an analytical solution. 

In this directory, Analytical/beam_modes.py provides some analytical results that can be used for comparison. Inputs to this file may need to be double checked for consistency against other parts of the verification procedure.

The BeamDyn folder contains BeamDyn inputs and outputs that are needed for the verification (no BeamDyn runs are needed here beyond the outputs already provided). 

In the Construction folder, first run "simple_construct.py" to generate 3DOF models for different cases. Then the script "verify_3dof.py" can be used to compare generate results that can be compared to the analytical solution. Near the top of "verify_3dof.py" there are user inputs to change between 3 different cases. The analytical solutions to compare against are in an excel sheet kept elsewhere. 

### BeamDyn Linear Regime Plot

The following generates a figure comparing BeamDyn to the low amplitude linearization of BeamDyn results.
```
cd OpenFAST/BeamDyn/linear_regime
python gen_bd_lin_check.py
source run_bd_lin.sh  # Assumes that BeamDyn driver is loaded
python linear_lims_plot.py
```
These commands generate the figure 'check_linear.png'.


## Nalu Wind Runs

To generate inputs:
```
cd nalu_wind
python utilities/gen_naluwind_inp_files.py
cd nalu_runs
mkdir job_list
cd job_list
find .. -name freq_* | sort -n > list_of_cases
sbatch job_submit.slurm
```

Note that in job_submit.slurm, the array of inputs is indexed from 1 to the number of lines in the file job_list.


### Collecting Nalu Wind results:

```
cd nalu_wind
python utilities/collect_naluwind_res.py 
```
This copies all of the input files and nc results files to a new folder and does summary calculations. The summary calculations are placed in "nalu_runs/ffaw3211/ffaw3211_stats.yaml" or an equivalent file path for runs in different folders. 
Summary plots can then be generated with the following:
```
cd ../python/Visualization/combined_results
cp ../../../nalu_wind/nalu_runs_2/ffaw3211/ffaw3211_stats.yaml .
mkdir Figures
python plot_damping.py
```

### Verification of Implementation Details

The mesh ramp can be manually checked with the runs in the folder of 'nalu_wind/cases/meshramp/'.

The time ramp can be manually checked with the runs in the folder 'nalu_wind/cases/timeramp/' plus the python script TO INSERT HERE that compares the forces.dat (unscaled) against the nc file (scaled).

Note that these simulations are not intended to have realistic CFD parameters and are just to check that the ramping appears to be working correctly not that the forces are correct.

## Misc Files 

This section contains a brief overview of miscellaneous files that were used while developing the main functionality of this repo. This should not be important at this point. 



#### python/DetermineRunSets

Script for generating a list of velocities of interest. Script also creates a plot to visualize what coverage of the frequency response for a linear system with constant external force would be covered.

These are probably not useful at this point. The script "freq_define.py" also calculates Reynolds numbers.

#### python/Verification

Scripts that could potentially be used to verify that time integration in Nalu-Wind is correct. The Unforced folder may be relevant (assuming loads_scale=0). These have not been fully used since we were still building confidence in the Nalu-Wind results from the CFD side. 

## Unsteady Aerodynamic Simulations

1. Clone this repo and the welib repo (https://github.com/gantech/welib) into the same folder. If you clone them in other folders, you will need to modify some file paths in the code. You will need to change to the sviv_2d branch of welib and follow the installation instructions on the README.md file for welib. 
2. Run a single case to determine input parameters. 
```
cd welib/welib/airfoils/examples/
python dynamic_stall_mhh.py
```
In the main portion of the previous python script, set "verify = False". Under "if not verify:", you can set the angle of attack (regenerates a new 3 DOF at the correct angle), the velocity, simulation time, and a time to slowly increase the loads over (ramp time). The time step for the output is set to 1e-3 is the "sviv_2d" function in this file. Running the file should generate figures for the time series response and the output file "ua_test.npz".
3. Copy the output file to do initial processing for PFF parameters (starting from the folder that contains clones of both repos):
```
cd sviv_2d/python/PFF/
cp ../../../welib/welib/airfoils/examples/ua_test.npz .
```
4. Set appropriate parameters in "peak_filter_fit.py" (in the folder that ua_test.npz was copied to). Use "import_flag = 1" to load the npz file (for CFD simulations, you can use "import_flag = 0" to load a netcdf file and do similar analysis). 
The main parameters that can be adjusted to get good results are: tstart (start signal processing at this time), nom_freq (nominal frequency, the peak frequency is identified close to this value, used to select frequency/mode of interest), and half_bandwidth_frac (Fraction of the peak frequency to use as the half bandwidth for the butterworth filter). 
The most important parameter is probably "half_bandwidth_frac": a narrow enough bandwidth is needed to filter out any other modes, however the narrower the bandwidth the longer the end effects are in the time domain. 
Lastly, you can set "remove_end" to eliminate data points from the end that are affected by end effects. This will generally be required for the automatic processing that occurs next, so this script should be used to determine what value to use. 
5. Run the script:
```
python peak_filter_fit.py 
```
6. Look at the figures that are produced. Figures with the word "verify" in the name correspond to examples. The other figures are for the loaded data set. Specifically "freq_damp_time.png" will show the frequency/damping as a function of time. Large fluctations near the end are expected end effects. "overview_pff_dir1.png" provides the time series for a given direction (0=x, 1=y, 2=theta). In these figures, you can compare the original signal to the identified peaks (circles) to understand the end effects.
7. Once the parameters are selected, set the parameters in "welib/welib/airfoils/examples/velocity_sweep.py" to match those selected.
8. Run the velocity sweep test. This script currently only loops over velocity, but updates the full angle of attack information, so should be relatively easy to insert a loop over angle of attack where marked. 
```
python velocity_sweep.py 
```
9. This script will create a yaml file summarizing the different runs. Copy this yaml file to the sviv_2d repo (starting from the top level directory again)
```
cd sviv_2d/python/Visualization/unsteady_aero/
cp ../../../../welib/welib/airfoils/examples/sweep_aoa50_ramp5.yaml .
```
10. Continuing from this folder, the results can be plotted with:
```
mkdir Figures
python plot_damping.py 
```
11. The figures are saved into the Figures folder. 




