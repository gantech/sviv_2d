"""
Make plots of damping factor and other key metrics for the paper
"""


import numpy as np

import yaml
from yaml.loader import SafeLoader 

import matplotlib.pyplot as plt



def load_data(collected_data):
    """
    Load the data dictionary
    """

    with open(collected_data) as f:
        struct_data = list(yaml.load_all(f, Loader=SafeLoader))
    
    data_dict = struct_data[0]
    
    
    # array that can be used to sort by freq
    freq_sort_inds = np.argsort(data_dict['Nom_Freq'])
    
    # Eliminate cases with nan outputs
    freq_sort_inds = freq_sort_inds[np.logical_not(np.isnan(np.array(data_dict['DispX_All_Amp'])[freq_sort_inds]))]

    return data_dict, freq_sort_inds


def plot_amp(data_dict, freq_sort_inds, mode_index, amp_units='m'):
    """
    Create plot of the amplitudes from PFF analysis

    Inputs: 
      mode_index - [0, 1, 2] corresponding to flap, edge or twist.
    """

    mode_name_list = ['flap', 'edge', 'twist']
    mode_name = mode_name_list[mode_index]


    vel = np.array(data_dict['velocity'])[freq_sort_inds]

    # first index correponds to a run, second to the reported sample amplitudes
    amp = np.array(data_dict[mode_name + '_mode_amp'])[freq_sort_inds, :]

   
    max_amp = np.abs(amp).max()

    ms = 2.5

    p = plt.plot(np.nan, np.nan)
    color = p[0].get_color()


    for i in range(vel.shape[0]):
        # Look at the amplitude in the primary direction of the mode
        modal_scale = np.array(data_dict['modeshapes_local'][i]).reshape(3,3)[mode_index, mode_index]
        modal_scale = np.abs(modal_scale)

        plt.plot(vel[i]*np.ones_like(amp[i, :]), np.abs(amp[i, :])*modal_scale, '.', 
                 markersize=ms, color=color)

    max_amp = max_amp * modal_scale

    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Amplitude [{}]'.format(amp_units))

    plt.ylim((0, max_amp*1.02))

    plt.tight_layout()

    plt.savefig('{}_amp_pff.png'.format(mode_name), dpi=300)
    plt.close()

def plot_damp(data_dict, freq_sort_inds, mode_index, amp_units='m'):
    """
    Create plot of the damping from PFF analysis

    Inputs: 
      mode_index - [0, 1, 2] corresponding to flap, edge or twist.
    """

    mode_name_list = ['flap', 'edge', 'twist']
    mode_name = mode_name_list[mode_index]


    vel = np.array(data_dict['velocity'])[freq_sort_inds]

    # first index correponds to a run, second to the reported sample amplitudes
    damp = np.array(data_dict[mode_name + '_mode_damp'])[freq_sort_inds, :]

   
    max_damp = damp.max()
    min_damp = damp.min()

    ms = 2.5

    p = plt.plot(np.nan, np.nan)
    color = p[0].get_color()


    for i in range(vel.shape[0]):

        plt.plot(vel[i]*np.ones_like(damp[i, :]), damp[i, :], '.', 
                 markersize=ms, color=color)

    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Damping Fraction Critical')

    plt.ylim((min_damp-0.01, max_damp+0.01))

    plt.tight_layout()

    plt.savefig('{}_damp_pff.png'.format(mode_name), dpi=300)
    plt.close()

def plot_freq(data_dict, freq_sort_inds, mode_index, amp_units='m'):
    """
    Create plot of the damping from PFF analysis

    Inputs: 
      mode_index - [0, 1, 2] corresponding to flap, edge or twist.
    """

    mode_name_list = ['flap', 'edge', 'twist']
    mode_name = mode_name_list[mode_index]


    vel = np.array(data_dict['velocity'])[freq_sort_inds]

    # first index correponds to a run, second to the reported sample amplitudes
    freq = np.array(data_dict[mode_name + '_mode_freq'])[freq_sort_inds, :]

    freq = freq / 2.0 / np.pi
   
    max_freq = freq.max()
    min_freq = freq.min()

    ms = 2.5

    p = plt.plot(np.nan, np.nan)
    color = p[0].get_color()


    for i in range(vel.shape[0]):

        plt.plot(vel[i]*np.ones_like(freq[i, :]), freq[i, :], '.', 
                 markersize=ms, color=color)

    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Damping Fraction Critical')

    plt.ylim((min_freq-0.01, max_freq+0.01))

    plt.tight_layout()

    plt.savefig('{}_freq_pff.png'.format(mode_name), dpi=300)
    plt.close()




if __name__=="__main__":

    ###########################
    # User inputs
    
    collected_data = 'ffaw3211_stats.yaml'

    plt.style.use('seaborn-v0_8-colorblind') 

    ###########################
    # Load Data

    data_dict, freq_sort_inds = load_data(collected_data)

    ###########################
    # Plot amplitude 

    plot_amp(data_dict, freq_sort_inds, 0,  amp_units='m')
    plot_amp(data_dict, freq_sort_inds, 1,  amp_units='m')
    plot_amp(data_dict, freq_sort_inds, 2, amp_units='rad')


    plot_damp(data_dict, freq_sort_inds, 0,  amp_units='m')
    plot_damp(data_dict, freq_sort_inds, 1,  amp_units='m')
    plot_damp(data_dict, freq_sort_inds, 2, amp_units='rad')


    plot_freq(data_dict, freq_sort_inds, 0,  amp_units='m')
    plot_freq(data_dict, freq_sort_inds, 1,  amp_units='m')
    plot_freq(data_dict, freq_sort_inds, 2, amp_units='rad')


