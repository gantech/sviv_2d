"""
Functions for processing smd runs to output statistics without plotting everything. 
"""


import numpy as np
import netCDF4 as nc
import yaml


def calc_nc_sum(filename, nominal_freq, dict):
    """
    Calculated summary statistics for an nc file and returns them
    """

    # load the data
    data = nc.Dataset(filename)

    time = np.array(data['time'][:])
    x = np.array(data['x'][:])
    xdot = np.array(data['xdot'][:])
    
    forces = np.array(data['f'][:]) 

    # Initialize values to calculate over 
    directions = ['X', 'Y', 'Theta']
    
    last_5_mask = time > (time[-1] - 5 / nominal_freq)
    all_mask = np.ones(time.size) == 1

    masks = [all_mask, last_5_mask]
    masks_name = ['All', 'Last5']

    data_list = [x, forces]
    data_list_name = ['Disp', 'F']

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
    filename = 'af_smd_deflloads.nc'
    output = 'stats.yaml'

    dict = {} # just initialize an empty dictionary

    # run the processing function
    calc_nc_sum(filename, 0.7, dict)
    calc_nc_sum(filename, 0.7, dict)

    # print the outputs to screen
    print(dict)

    with open(output, 'w') as outfile:
        yaml.dump(dict, outfile)
