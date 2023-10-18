"""
Make a 2D Contour plot of Angle of Attack versus Velocity for the damping value
"""

import numpy as np

import yaml
from yaml.loader import SafeLoader 

import matplotlib.pyplot as plt

import plot_damping as pdamp


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
    damp = pdamp.load_to_list(data_dict, mode_name + '_mode_damp')
   
    max_damp = pdamp.get_max_list(damp)
    min_damp = pdamp.get_min_list(damp)
    mean_damp = pdamp.get_mean_list(damp)

    ms = 2.5*4

    Nout = np.array([x.shape[0] for x in damp]).max() # use for setting how light the lightest point is

    for i in range(vel.shape[0]):

        plt.scatter(vel[i]*np.ones_like(damp[i]), damp[i], 
                 c=-(np.array(range(damp[i].shape[0]))-damp[i].shape[0]), 
                 s=ms, vmin=1, vmax=1.2*Nout)

    
    plt.scatter(vel, mean_damp, 
             c='r', 
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

    plt.show()
    plt.close()
    
    
def load_all_damp(aoa_list, mode_index=1, Nvel=11):
    

    mode_name_list = ['flap', 'edge', 'twist']
    mode_name = mode_name_list[mode_index]
    
    
    damp_mat = np.zeros((len(aoa_list), Nvel))
    
    aoa_vec = np.array(aoa_list)

    for i in range(len(aoa_list)):
        
        collected_data = 'sweep_aoa{}_ramp5.yaml'.format(aoa_list[i]) 

        data_dict = pdamp.load_data(collected_data)
        
        
        vel = np.array(data_dict['velocity'])
        
        damp = pdamp.load_to_list(data_dict, mode_name + '_mode_damp')

        
        mean_damp = pdamp.get_mean_list(damp)
        # print(mean_damp)
        # print(damp_mat.shape)
        # print(i)
        
        damp_mat[i, :] = mean_damp
    
    return vel, aoa_vec, damp_mat
    
if __name__=="__main__":

    ###########################
    # User inputs
    
    # Also ran aoa = 30, but did not appear converged / steady state
    aoa_list = [35, 40, 45, 50, 55, 60, 65]
    
    
    vel, aoa_vec, damp = load_all_damp(aoa_list, mode_index=1, Nvel=11)
    
    # print(damp.shape)
    # print(damp)
    
    levels = [-0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02]
    
    plt.contourf(vel, aoa_vec, damp, levels)
    
    
    plt.xlabel('Inflow Velocity [m/s]')
    plt.ylabel('Angle of Attack [deg]')
    plt.tight_layout()
    
    clb = plt.colorbar()
    clb.set_label('Fraction Critical Damping')
    
    plt.grid()
    

    plt.savefig('Figures/contourf_damping_edge.png', dpi=300)
    plt.show()
    
    # aoa = 30
    # for aoa in [30, 35, 40, 45, 50, 55, 60, 65]:
    for aoa in [50]:
        
        # collected_data = 'initial_sweep_aoa50.yaml'
        # collected_data = 'sweep_aoa50.yaml' # 0 ramp time
        collected_data = 'sweep_aoa{}_ramp5.yaml'.format(aoa) # ramp over 5.0 s
    
        plt.style.use('seaborn-colorblind') 
    
        ###########################
        # Load Data
    
        data_dict = pdamp.load_data(collected_data)
    
        ###########################
        # Plot amplitude 
    
        plot_damp(data_dict, 0,  amp_units='m', aoa=aoa)
        plot_damp(data_dict, 1,  amp_units='m', aoa=aoa)
        plot_damp(data_dict, 2, amp_units='rad', aoa=aoa)
    


