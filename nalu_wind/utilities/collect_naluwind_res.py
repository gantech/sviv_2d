"""
Script for processing naluwind outputs and collecting results into a yaml file

Assumes for folder path that this script is called from the top level of nalu-wind folder
"""

import yaml, glob
from yaml.loader import SafeLoader 

from pathlib import Path
import os
import numpy as np

import sys
sys.path.append('../python/Visualization/')
sys.path.append('../python/PFF')

import calc_stats as cstats
import peak_filter_fit as pff


def collect_folders(run_folder='nalu_runs/ffaw3211', output_name='ffaw3211_stats.yaml', copy_nc_folder='./collect_nc'):
    """
    Run folder should be a relative path, otherwise the copying will probably do something weird. 
    copy_nc_folder - should work as an absolute path
    """

    yaml3dof = 'chord_3dof.yaml'
    ncfilename = 'af_smd_deflloads.nc'
    input_name = '*.yaml'

    # Create Dictionary / Initialization
    dict = {}


    ##### List of Structures (AOA varies with structure potentially, or mode shape)
    p = Path(run_folder)
    struct_dirs = [f for f in p.iterdir() if f.is_dir()]

    for struct in struct_dirs:

        struct_ind = int(struct.parts[-1][10:])

        # Load yaml for this structure
        path_3dof = os.path.join(struct, yaml3dof)

        # Copy the yaml file for later analysis to a folder that will be saved
        copy_path_file_3dof = os.path.join(copy_nc_folder, path_3dof)
        Path(os.path.join(copy_nc_folder, struct)).mkdir(parents=True, exist_ok=True)

        print(path_3dof)
        print(copy_path_file_3dof)
        os.system('cp {} {}'.format(path_3dof, copy_path_file_3dof))

        # Load the yaml
        with open(path_3dof) as f:
            struct_data = list(yaml.load_all(f, Loader=SafeLoader))
        Tmat = np.array(struct_data[0]['force_transform_matrix']).reshape(3,3)
        mode_shapes = np.array(struct_data[0]['phi_matrix']).reshape(3,3)
        aoa = struct_data[0]['angle']

        # Identify frequency runs for this structure
        p = Path(struct)
        freq_dirs = [f for f in p.iterdir() if f.is_dir()]

        for freq_folder in freq_dirs:

            freq = float(freq_folder.parts[-1][5:])

            # Actually calculate the response statistics here:
            path_file_nc = os.path.join(freq_folder, ncfilename)

            # Copy the nc file for later analysis to a folder that will be saved
            copy_path_file_nc = os.path.join(copy_nc_folder, freq_folder)
            Path(copy_path_file_nc).mkdir(parents=True, exist_ok=True)
            os.system('cp {} {}'.format(path_file_nc, copy_path_file_nc))

            path_file_input = os.path.join(freq_folder, input_name)
            os.system('cp {} {}'.format(path_file_input, copy_path_file_nc))

            # Pull velocity from the template input
            with open(glob.glob(path_file_input)[0]) as f:
                tfile = yaml.load(f,Loader=yaml.UnsafeLoader)
            velocity = tfile['realms'][0]['boundary_conditions'][1]['inflow_user_data']['velocity'][0]

            # Calculate some stats now and add to dictionary
            cstats.calc_nc_sum(path_file_nc, freq, dict, force_trans=Tmat, aoa=aoa, 
                               struct_ind=struct_ind, mode_shapes=mode_shapes, velocity=velocity)

    # Save the output to a file
    output = os.path.join(run_folder, output_name)

    with open(output, 'w') as outfile:
        yaml.dump(dict, outfile)


if __name__=="__main__":

    # Call functios to do the post processing
    collect_folders(run_folder='nalu_runs_2/ffaw3211')

