"""
Script for collecting results of BeamDyn distributed runs and making a plot
to verify the linear regime of the beam model. 

This script does not actually run the BeamDyn models. 
"""

import numpy as np

import os
from pathlib import Path

from pyFAST.input_output import FASTOutputFile

import matplotlib.pyplot as plt

# import matplotlib as mpl
# mpl.rc('text', usetex=True)


############
# Define Parameters of the Search and Load

# # All cases considered
# top_folders = ['flap', 'edge', 'twist', 'flaptwist', 'edgetwist']

# Consider just the 3 cases with a single loading
top_folders = ['flap', 'edge', 'twist']

output_file = 'bd_driver.out'

disp_keys = ['TipTDxr_[m]', 'TipTDyr_[m]', 'TipRDzr_[-]', 'TipTDxr_[m]', 'TipTDyr_[m]'] 


############
# Start loop over directories loading data

load_all = len(top_folders) * [None]
disp_all = len(top_folders) * [None]

for case_ind in range(len(top_folders)):

    
    p = Path(top_folders[case_ind])

    run_directories = [f for f in p.iterdir() if f.is_dir()]

    case_load = np.zeros(len(run_directories))
    case_disp = np.zeros(len(run_directories))

    
    for path_ind in range(len(run_directories)):

        curr_path = run_directories[path_ind]
    
        case_load[path_ind] = float(curr_path.parts[-1][4:])
    
        filename = os.path.join(curr_path, output_file)

        # print(filename)
        df = FASTOutputFile(filename).toDataFrame()

        case_disp[path_ind] = df[disp_keys[case_ind]].to_numpy()[-1]

    # Sort Load Levels and use indices from results to sort the displacements

    # Save data to lists
    indices = np.argsort(case_load)

    disp_all[case_ind] = case_disp[indices]
    load_all[case_ind] = case_load[indices]


############
# Plot Results

plt.style.use('seaborn-v0_8-colorblind') 

disp_convert = [1, 1, 180/np.pi, 1, 1]

label_names = ['Flap [m]', 'Edge [m]', 'Twist [deg]', 'Flap [m]', 'Edge [m]']

max_y = 0.0
min_y = 0.0

min_x = 0.0

for case_ind in range(len(top_folders)):

    p = plt.plot(load_all[case_ind], disp_all[case_ind]*disp_convert[case_ind],
              'o', label=label_names[case_ind],
              markersize=6, fillstyle='none')

    max_y = np.maximum((disp_all[case_ind]*disp_convert[case_ind]).max(), max_y)
    min_y = np.minimum((disp_all[case_ind]*disp_convert[case_ind]).min(), min_y)

    minind_pos = np.argmin(np.abs(load_all[case_ind] - 100))
    minind_neg = np.argmin(np.abs(load_all[case_ind] + 100))

    flexibility = (disp_all[case_ind][minind_pos] - disp_all[case_ind][minind_neg]) \
                    / (load_all[case_ind][minind_pos] - load_all[case_ind][minind_neg])

    lin_disp = flexibility*load_all[case_ind]

    plt.plot(load_all[case_ind], lin_disp*disp_convert[case_ind], 
              '-', label='Linear ' + label_names[case_ind], color=p[0].get_color())

    max_y = np.maximum((lin_disp*disp_convert[case_ind]).max(), max_y)
    min_y = np.minimum((lin_disp*disp_convert[case_ind]).min(), min_y)

    min_x = np.minimum(load_all[case_ind].min(), min_x)

plt.legend()
plt.xlabel('Load [N/m or Nm/m]')
plt.ylabel('Tip Displacement [m or deg]')

plt.xlim((min_x*1.05, load_all[0][-1]*1.05)) 
plt.ylim((min_y*1.05, max_y*1.05))

plt.savefig('check_linear.png')
plt.close()
