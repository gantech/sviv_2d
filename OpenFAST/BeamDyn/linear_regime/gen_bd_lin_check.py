"""
Generate beamdyn inputs to check the linear regime of BeamDyn
"""


import numpy as np

import os
from pathlib import Path

from pyFAST.input_output import FASTInputFile


filename = 'bd_driver_template.inp' # Template file name

bd_name = 'bd_driver.inp' # BeamDyn input for the specific case has this name in the given folder.


top_folders = ['flap', 'edge', 'twist']

load_keys = ['DistrLoad(1)', 'DistrLoad(2)', 'DistrLoad(6)']

load_levels = np.array([-2.5e4, -2e4, -1.5e4, -1e4, -.5e4, -100, 100, .5e4, 1e4, 1.5e4, 2.0e4, 2.5e4])

# Loop over top level folders for load directions
for case_ind in range(len(top_folders)):

    p = Path(top_folders[case_ind])

    for load_ind in range(load_levels.shape[0]):
    
        load = np.round(load_levels[load_ind])

        folder_path = os.path.join(p, 'dist'+str(load))

        Path(folder_path).mkdir(parents=True, exist_ok=True)

        file_path = os.path.join(folder_path, bd_name)

        # Load Template
        f = FASTInputFile(filename)
        f[load_keys[case_ind]] = load

        f.write(file_path)

