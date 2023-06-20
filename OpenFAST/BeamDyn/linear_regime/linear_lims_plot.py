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

############
# Define Parameters of the Search and Load


top_folders = ['flap', 'edge', 'twist']

output_file = 'bd_driver.out'

disp_keys = ['TipTDxr_[m]', 'TipTDyr_[m]', 'TipRDzr_[-]'] # add keys for edge and twist here.



############
# Start loop over directories loading data

load_all = 3 * [None]
disp_all = 3 * [None]

for case_ind in range(len(top_folders)):

    
    p = Path(top_folders[case_ind])

    run_directories = [f for f in p.iterdir() if f.is_dir()]

    case_load = np.zeros(len(run_directories))
    case_disp = np.zeros(len(run_directories))

    
    for path_ind in range(len(run_directories)):

        curr_path = run_directories[path_ind]
    
        case_load[path_ind] = float(curr_path.parts[-1][4:])
    
        filename = os.path.join(curr_path, output_file)
        df = FASTOutputFile(filename).toDataFrame()

        case_disp[path_ind] = df[disp_keys[case_ind]].to_numpy()[-1]

    # Sort Load Levels and use indices from results to sort the displacements

    # Save data to lists
    indices = np.argsort(case_load)

    disp_all[case_ind] = case_disp[indices]
    load_all[case_ind] = case_load[indices]


############
# Plot Results

for case_ind in range(len(top_folders)):
    plt.plot(load_all[case_ind], disp_all[case_ind], '.', label=top_folders[case_ind] + ' ' + disp_keys[case_ind])

    lin_disp = disp_all[case_ind][0] / load_all[case_ind][0]*load_all[case_ind]

    plt.plot(load_all[case_ind], lin_disp, '-', label=top_folders[case_ind] + ' ' + disp_keys[case_ind])


plt.legend()
plt.xlabel('Load [N/m or Nm/m]')
plt.ylabel('Tip Displacement [m or rad]')

plt.savefig('check_linear.png')
plt.close()
