# coding: utf-8
from pyFAST.input_output import FASTOutputFile
import pandas as pd

import numpy as np

# import yaml
# from yaml.loader import SafeLoader

import sys
sys.path.append('..')
import construct_utils as cutils


##### Define Comparison Cases

case_names = ['Flap', 'Edge', 'Twist']
static_bd = ['bd_driver_flap.out', 'bd_driver_edge.out', 'bd_driver_twist.out']
load_dir_ind = [0, 1, 5] # Flap, Edge, Twist

##### Stiffness Matrix from Dynamic Outputs

Mfull,Kfull,node_coords,quad_coords = cutils.load_M_K_nodes('bd_driver.BD.sum.yaml')

# # Load K
# with open('bd_driver.BD.sum.yaml') as f:
#     data = list(yaml.load_all(f, Loader=SafeLoader))
# 
# dict_data = data[-1]
# Kfull = np.array(dict_data['K_BD']) # This uses BD coordinate system, but the tip displacements may be in IEC. May be same here.
Kfull = Kfull[6:, 6:] # Apply fixed Boundary Conditions


######
# Insert Loop over 3 cases
# for ind in [2]:    
for ind in range(3):

    print('Evaluating case: {}'.format(case_names[ind]))
    
    # Load A static Input:
    df_flap = FASTOutputFile(static_bd[ind]).toDataFrame()
    # print(df_flap.keys())
    
    # Generate a static analysis.
    F = np.zeros(60)
    
    # Set the Middle node = node 6, but have eliminated node 1 already with BC. Therefore should be DOFS 24-29
    F[24 + load_dir_ind[ind]] = 1000
    
    Uflap_static = np.linalg.solve(Kfull, F)

    # print(Uflap_static[-6:])
    
    print(df_flap[['TipTDxr_[m]', 'TipTDyr_[m]', 'TipTDzr_[m]', 'TipRDxr_[-]', 'TipRDyr_[-]', 'TipRDzr_[-]']].iloc[-1])
    
    
    flap_np = df_flap[['TipTDxr_[m]', 'TipTDyr_[m]', 'TipTDzr_[m]', 'TipRDxr_[-]', 'TipRDyr_[-]', 'TipRDzr_[-]']].to_numpy()
    
    print('Fractional Error:')
    print(np.abs(flap_np[-1] - Uflap_static[-6:])/np.abs(flap_np[-1]))
    print('')    

    nodal_disp = np.zeros_like(node_coords[:, 2])
    # First entry is 0 for BC
    nodal_disp[1:] = Uflap_static[load_dir_ind[ind]::6]

    cutils.plot_nodal_field(node_coords[:, 2], nodal_disp, '{}.png'.format(case_names[ind]),
                            ylabel=case_names[ind])
