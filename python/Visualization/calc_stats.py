"""
Functions for processing smd runs to output statistics without plotting everything. 
"""


import numpy as np
import netCDF4 as nc
import yaml
from yaml.loader import SafeLoader 

import sys
sys.path.append('../PFF')
import peak_filter_fit as pff


def pff_summary(t, x, forces, mode_shapes, nom_freq, dict,
                half_bandwidth_frac=0.05, tstart=10, remove_end=7, reportnum=20):
    """
    Function to conduct PFF analysis on nc file data

    Inputs:
      t - time series
      x - physical coordinates, displacements
      forces - physical coordinates (after conversion to 3 DOF model)
      mode_shapes - 3x3 matrix of mode shapes. Columns are modes=[Flap, Edge, Twist], rows are
                    flap, edge, twist contributions of the given mode. 

      tstart - start of time analysis, probably want to eliminate the first several cycles
      remove_end - remove a number of points off the end of the pff analysis, 
                   for multiharmonic signals this is needed to eliminate filtering end effects
      reportnum - number of values from PFF to report
      

    Outputs: 
      - Various values to save summarizing PFF results
    """

    if np.isnan(x).any():
        create_append_dict(dict,  'flap_mode_amp', (np.ones(reportnum)*np.nan).tolist())
        create_append_dict(dict,  'edge_mode_amp', (np.ones(reportnum)*np.nan).tolist())
        create_append_dict(dict, 'twist_mode_amp', (np.ones(reportnum)*np.nan).tolist())

        create_append_dict(dict,  'flap_mode_damp', (np.ones(reportnum)*np.nan).tolist())
        create_append_dict(dict,  'edge_mode_damp', (np.ones(reportnum)*np.nan).tolist())
        create_append_dict(dict, 'twist_mode_damp', (np.ones(reportnum)*np.nan).tolist())

        create_append_dict(dict,  'flap_mode_freq', (np.ones(reportnum)*np.nan).tolist())
        create_append_dict(dict,  'edge_mode_freq', (np.ones(reportnum)*np.nan).tolist())
        create_append_dict(dict, 'twist_mode_freq', (np.ones(reportnum)*np.nan).tolist())

        create_append_dict(dict,  'flap_mode_f_amp', (np.ones(reportnum)*np.nan).tolist())
        create_append_dict(dict,  'edge_mode_f_amp', (np.ones(reportnum)*np.nan).tolist())
        create_append_dict(dict, 'twist_mode_f_amp', (np.ones(reportnum)*np.nan).tolist())

        create_append_dict(dict,  'flap_mode_f_freq', (np.ones(reportnum)*np.nan).tolist())
        create_append_dict(dict,  'edge_mode_f_freq', (np.ones(reportnum)*np.nan).tolist())
        create_append_dict(dict, 'twist_mode_f_freq', (np.ones(reportnum)*np.nan).tolist())

    else:    
    
        # Convert to the modal domain
        modal_q = (np.linalg.inv(mode_shapes) @ x.T).T # use inv of modes to convert to modal domain.
        modal_f = (mode_shapes @ forces.T).T
    
        freq_rad_s_q, damp_frac_crit_q, report_t_q, report_amp_q, intermediate_data_q = \
               pff.pff_analysis(t, modal_q, nom_freq, tstart, half_bandwidth_frac, remove_end=remove_end)
    
    
        freq_rad_s_f, damp_frac_crit_f, report_t_f, report_amp_f, intermediate_data_f = \
                  pff.pff_analysis(t, modal_f, nom_freq, tstart, half_bandwidth_frac, remove_end=remove_end)
    
    
        create_append_dict(dict, 'flap_mode_amp', report_amp_q[0][-reportnum:].tolist())
        create_append_dict(dict, 'edge_mode_amp', report_amp_q[1][-reportnum:].tolist())
        create_append_dict(dict, 'twist_mode_amp', report_amp_q[2][-reportnum:].tolist())

        create_append_dict(dict,  'flap_mode_damp', damp_frac_crit_q[0][-reportnum:].tolist())
        create_append_dict(dict,  'edge_mode_damp', damp_frac_crit_q[1][-reportnum:].tolist())
        create_append_dict(dict, 'twist_mode_damp', damp_frac_crit_q[2][-reportnum:].tolist())

        create_append_dict(dict,  'flap_mode_freq_rad_s', freq_rad_s_q[0][-reportnum:].tolist())
        create_append_dict(dict,  'edge_mode_freq_rad_s', freq_rad_s_q[1][-reportnum:].tolist())
        create_append_dict(dict, 'twist_mode_freq_rad_s', freq_rad_s_q[2][-reportnum:].tolist())

        create_append_dict(dict,  'flap_mode_f_amp', report_amp_f[0][-reportnum:].tolist())
        create_append_dict(dict,  'edge_mode_f_amp', report_amp_f[1][-reportnum:].tolist())
        create_append_dict(dict, 'twist_mode_f_amp', report_amp_f[2][-reportnum:].tolist())

        create_append_dict(dict,  'flap_mode_f_freq_rad_s', freq_rad_s_f[0][-reportnum:].tolist())
        create_append_dict(dict,  'edge_mode_f_freq_rad_s', freq_rad_s_f[1][-reportnum:].tolist())
        create_append_dict(dict, 'twist_mode_f_freq_rad_s', freq_rad_s_f[2][-reportnum:].tolist())

    print('Several inputs for PFF analysis need to be decided.')




def calc_nc_sum(filename, nominal_freq, dict, force_trans=np.eye(3), aoa=-310, struct_ind=0, 
                mode_shapes=np.eye(3), velocity=-1e12, 
                half_bandwidth_frac=0.05, tstart=10, remove_end=7, reportnum=20):
    """
    Calculated summary statistics for an nc file and returns them
    """

    # load the data
    data = nc.Dataset(filename)

    time = np.array(data['time'][:])
    x = np.array(data['x'][:])
    xdot = np.array(data['xdot'][:])
    
    forces = (force_trans @ np.array(data['f'][:]).T ).T

    # Initialize values to calculate over 
    directions = ['X', 'Y', 'Theta']
    
    last_5_mask = time > (time[-1] - 5 / nominal_freq)
    all_mask = np.ones(time.size) == 1

    masks = [all_mask, last_5_mask]
    masks_name = ['All', 'Last5']

    data_list = [x, forces]
    data_list_name = ['Disp', 'F']

    create_append_dict(dict, 'struct_ind', struct_ind)
    create_append_dict(dict, 'aoa', aoa)
    create_append_dict(dict, 'velocity', velocity)

    # Rotate the mode shape back to zero deg AOA and save
    aoa_rad = aoa * np.pi/180.0
    zeroaoa_mode = np.array([[np.sin(aoa_rad), np.cos(aoa_rad), 0], 
                             [np.cos(aoa_rad), -np.sin(aoa_rad), 0], 
                             [0,0,-1]]).T \
                   @ mode_shapes

    create_append_dict(dict, 'modeshapes_local', zeroaoa_mode.reshape(-1).tolist())
    create_append_dict(dict, 'mode_shapes', mode_shapes.reshape(-1).tolist())

    # Calculate lots of different statistics
    for dir_ind in range(len(directions)):
        for mask_ind in range(len(masks)):
            for data_ind in range(len(data_list)):

                # Define the name
                name = data_list_name[data_ind] \
                        + directions[dir_ind] \
                        + '_' \
                        + masks_name[mask_ind]

                # Data to analyze
                data_curr = data_list[data_ind][:, dir_ind][masks[mask_ind]]

                # Statistics
                amp = (data_curr.max() - data_curr.min())/2.0
                mean = (data_curr.max() + data_curr.min())/2.0

                # Save outputs into a list or something
                create_append_dict(dict, name + '_Amp', float(amp))
                create_append_dict(dict, name + '_Mean', float(mean))

    # Add inputs to the dictionary
    create_append_dict(dict, 'Nom_Freq', nominal_freq)
    create_append_dict(dict, 'Nsteps', time.shape[0])

    pff_summary(time, x, forces, mode_shapes, nominal_freq, dict, 
                half_bandwidth_frac=half_bandwidth_frac, tstart=tstart,
                remove_end=remove_end, reportnum=reportnum)

def create_append_dict(dict, key, val):
    """
    Quick function to either append data to list or start a list for the key
    """

    if key in dict:
        dict[key] += [val]
    else:
        dict[key] = [val]

if __name__=="__main__":
    
    # Load the nc file
    filename = './single_results/af_smd_deflloads.nc'
    output = './single_results/stats.yaml'

    dict = {} # just initialize an empty dictionary

    ##### Load T matrxi from specific 3 DOF model
    yaml3dof = './single_results/chord_3dof.yaml'

    with open(yaml3dof) as f:
        struct_data = list(yaml.load_all(f, Loader=SafeLoader))
    Tmat = np.array(struct_data[0]['force_transform_matrix']).reshape(3,3)

    # second argument here is the nominal frequency based on the folder name
    # run the processing function
    calc_nc_sum(filename, 10, dict, force_trans=Tmat)
    calc_nc_sum(filename, 0.7, dict, force_trans=Tmat)

    # print the outputs to screen
    print(dict)

    with open(output, 'w') as outfile:
        yaml.dump(dict, outfile)
