"""
Plot some summary data that is collected from many nalu wind runs.
"""

import numpy as np

import yaml
from yaml.loader import SafeLoader 

import matplotlib.pyplot as plt

###########################
# User inputs

collected_data = 'ffaw3211_stats.yaml'




###########################
# Load Summary matrix

with open(collected_data) as f:
    struct_data = list(yaml.load_all(f, Loader=SafeLoader))

data_dict = struct_data[0]


# array that can be used to sort by freq
freq_sort_inds = np.argsort(data_dict['Nom_Freq'])

# Eliminate cases with nan outputs
freq_sort_inds = freq_sort_inds[np.logical_not(np.isnan(np.array(data_dict['DispX_All_Amp'])[freq_sort_inds]))]


###########################
# Subplot with force and displacement amplitudes, need 3 for 3 directions


directions = ['X', 'Y', 'Theta']
masks_name = ['All', 'Last5']
data_list_name = ['Disp', 'F']

def plot_summary(dir_ind, mask_ind, data_dict, freq_sort_inds, directions, masks_name, data_list_name):
    """
    Plot a case of the summary
    """
    # Define the name
    name = directions[dir_ind] \
            + '_' \
            + masks_name[mask_ind] \
            + '_Amp'
    
    freq_plot =  np.array(data_dict['Nom_Freq'])[freq_sort_inds]
    disp_plot =  np.array(data_dict[data_list_name[0] + name])[freq_sort_inds]
    force_plot = np.array(data_dict[data_list_name[1] + name])[freq_sort_inds]
    
    
    # print(freq_plot)
    # print(disp_plot)
    # print(force_plot)
    # print(name)
    
    fig, axs = plt.subplots(2)
    
    xlims = (freq_plot[0]-0.02, freq_plot[-1]+0.02)
    
    axs[0].plot(freq_plot, disp_plot, 'o', label=directions[dir_ind])
    axs[0].set_xlim(xlims)
    axs[0].set_ylabel('Displacement Amp')
    axs[0].set_xlabel('Nominal Excitation Frequency [Hz]')
    axs[0].legend()
    axs[0].set_title(name)
    
    axs[1].plot(freq_plot, force_plot, 'o', label='Force/Moment')
    axs[1].set_xlim(xlims)
    axs[1].set_ylabel('Force / Moment')
    axs[1].set_xlabel('Nominal Excitation Frequency [Hz]')
    axs[1].legend()
    
    fig.savefig(name+'.png')
    plt.close(fig)

for dir_ind in range(len(directions)):
    for mask_ind in range(len(masks_name)):
        plot_summary(dir_ind, mask_ind, data_dict, freq_sort_inds, directions, masks_name, data_list_name)

