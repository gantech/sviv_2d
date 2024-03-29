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
    
    return data_dict

def load_to_list(data_dict, key):
    """
    Load a dictionary item into a list of numpy arrays since the size may be irregular
    """

    
    val = [np.array(x) for x in data_dict[key]]

    return val

def get_max_list(val):

    max_val = np.array([x.max() for x in val if x.shape[0] != 0]).max()

    return max_val

def get_min_list(val):

    min_val = np.array([x.min() for x in val if x.shape[0] != 0]).min()

    return min_val

def get_mean_list(val):

    mean_val = np.array([x.mean() for x in val if x.shape[0] != 0])

    return mean_val

def plot_amp(data_dict, mode_index, amp_units='m', aoa=-100):
    """
    Create plot of the amplitudes from PFF analysis

    Inputs: 
      mode_index - [0, 1, 2] corresponding to flap, edge or twist.
    """

    mode_name_list = ['flap', 'edge', 'twist']
    mode_name = mode_name_list[mode_index]


    vel = np.array(data_dict['velocity'])

    # for item in data_dict[mode_name + '_mode_amp']:
    #     print(len(item))

    # first index correponds to a run, second to the reported sample amplitudes
    amp = load_to_list(data_dict, mode_name + '_mode_amp')

    max_amp = get_max_list(amp)

    ms = 2.5*4

    Nout = np.array([x.shape[0] for x in amp]).max() # use for setting how light the lightest point is

    for i in range(vel.shape[0]):
        # Look at the amplitude in the primary direction of the mode
        if mode_index == 0 or mode_index == 1:
            # Total Translational displacement
            modal_scale = np.array(data_dict['mode_shapes'][i]).reshape(3,3)[1:3, mode_index]
            modal_scale = np.sqrt(np.sum(modal_scale**2))

        else:
            # Just twist component for the third mode
            modal_scale = np.array(data_dict['mode_shapes'][i]).reshape(3,3)[mode_index, mode_index]
            modal_scale = np.abs(modal_scale)

        plt.scatter(vel[i]*np.ones_like(amp[i]), np.abs(amp[i])*modal_scale, 
                 c=-(np.array(range(amp[i].shape[0]))-amp[i].shape[0]), 
                 s=ms, vmin=1, vmax=1.2*Nout)

        plt.gray()

    max_amp = max_amp * modal_scale

    plt.xlabel('Inflow Velocity [m/s]')
    plt.ylabel('Amplitude [{}]'.format(amp_units))
    plt.title('Angle of Attack = {} Degrees'.format(aoa))

    plt.ylim((0, max_amp*1.02))

    plt.tight_layout()

    plt.savefig('Figures/aoa{}/{}_amp_pff.png'.format(aoa, mode_name), dpi=300)
    plt.show()
    plt.close()

def plot_damp(data_dict, mode_index, amp_units='m', aoa=-360):
    """
    Create plot of the damping from PFF analysis

    Inputs: 
      mode_index - [0, 1, 2] corresponding to flap, edge or twist.
    """

    mode_name_list = ['flap', 'edge', 'twist']
    mode_name = mode_name_list[mode_index]


    vel = np.array(data_dict['velocity'])

    # first index correponds to a run, second to the reported sample amplitudes
    damp = load_to_list(data_dict, mode_name + '_mode_damp')
   
    max_damp = get_max_list(damp)
    min_damp = get_min_list(damp)

    ms = 2.5*4

    Nout = np.array([x.shape[0] for x in damp]).max() # use for setting how light the lightest point is

    for i in range(vel.shape[0]):

        plt.scatter(vel[i]*np.ones_like(damp[i]), damp[i], 
                 c=-(np.array(range(damp[i].shape[0]))-damp[i].shape[0]), 
                 s=ms, vmin=1, vmax=1.2*Nout)

    plt.xlabel('Inflow Velocity [m/s]')
    plt.ylabel('Damping Fraction Critical')
    plt.title('Angle of Attack = {} Degrees'.format(aoa))

    plt.ylim((min_damp-0.01, max_damp+0.01))

    plt.tight_layout()
    
    ax = plt.gca()
    xlim = ax.get_xlim()
    plt.plot([xlim[0], xlim[1]], [0, 0], 'k', linewidth=1.0)
    ax.set_xlim(xlim)

    plt.savefig('Figures/aoa{}/{}_damp_pff.png'.format(aoa,mode_name), dpi=300)
    plt.show()
    plt.close()

def plot_freq(data_dict, mode_index, amp_units='m', aoa=-360):
    """
    Create plot of the damping from PFF analysis

    Inputs: 
      mode_index - [0, 1, 2] corresponding to flap, edge or twist.
    """

    mode_name_list = ['flap', 'edge', 'twist']
    mode_name = mode_name_list[mode_index]


    vel = np.array(data_dict['velocity'])

    # first index correponds to a run, second to the reported sample amplitudes
    freq = load_to_list(data_dict, mode_name + '_mode_freq_rad_s')

    freq = [x / 2.0 / np.pi for x in freq]
   
    max_freq = get_max_list(freq)
    min_freq = get_min_list(freq)

    ms = 2.5*4

    Nout = np.array([x.shape[0] for x in freq]).max() # use for setting how light the lightest point is

    for i in range(vel.shape[0]):

        plt.scatter(vel[i]*np.ones_like(freq[i]), freq[i], 
                 c=-(np.array(range(freq[i].shape[0]))-freq[i].shape[0]), 
                 s=ms, vmin=1, vmax=1.2*Nout)

    plt.xlabel('Inflow Velocity [m/s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Angle of Attack = {} Degrees'.format(aoa))

    plt.ylim((min_freq-0.01, max_freq+0.01))

    plt.tight_layout()

    plt.savefig('Figures/aoa{}/{}_freq_pff.png'.format(aoa, mode_name), dpi=300)
    plt.show()
    plt.close()




if __name__=="__main__":

    ###########################
    # User inputs
    
    # aoa = 30
    for aoa in [30, 35, 40, 45, 50, 55, 60, 65]:
        # collected_data = 'initial_sweep_aoa50.yaml'
        # collected_data = 'sweep_aoa50.yaml' # 0 ramp time
        collected_data = 'sweep_aoa{}_ramp5.yaml'.format(aoa) # ramp over 5.0 s
    
        plt.style.use('seaborn-colorblind') 
    
        ###########################
        # Load Data
    
        data_dict = load_data(collected_data)
    
        ###########################
        # Plot amplitude 
    
        plot_amp(data_dict, 0,  amp_units='m', aoa=aoa)
        plot_amp(data_dict, 1,  amp_units='m', aoa=aoa)
        plot_amp(data_dict, 2, amp_units='rad', aoa=aoa)
    
        plot_damp(data_dict, 0,  amp_units='m', aoa=aoa)
        plot_damp(data_dict, 1,  amp_units='m', aoa=aoa)
        plot_damp(data_dict, 2, amp_units='rad', aoa=aoa)
    
        plot_freq(data_dict, 0,  amp_units='m', aoa=aoa)
        plot_freq(data_dict, 1,  amp_units='m', aoa=aoa)
        plot_freq(data_dict, 2, amp_units='rad', aoa=aoa)


