# coding: utf-8
from pyFAST.input_output import FASTOutputFile
import pandas as pd

import numpy as np

import yaml
from yaml.loader import SafeLoader



##### Function for Constructing Local Flexibility Matrix

def construct_flexibility(Kfull, applied_mag):
    """
    Construct the flexibility matrix by applying load in 3 directions and calculating
    the corresponding displacements in those directions
    """

    pos_ind = 5 # Output Node/Quad Point of interest (numbered starting from 1 after eliminating B.C.s)
    
    load_dir_ind = [0, 1, 5] # Flap, Edge, Twist
    
    local_flexibility = np.zeros((3,3))
    
    for dind in range(len(load_dir_ind)):
    
        F = np.zeros(Kfull.shape[0])
    
        F[(pos_ind-1)*6+load_dir_ind[dind]] = applied_mag
    
        U_static = np.linalg.solve(Kfull, F)
    
        for rind in range(len(load_dir_ind)):
    
            local_flexibility[rind, dind] = U_static[(pos_ind-1)*6+load_dir_ind[rind]]

    return local_flexibility 
    

##### Define Comparison Cases

# case_names = ['Flap', 'Edge', 'Twist']
# static_bd = ['bd_driver_flap.out', 'bd_driver_edge.out', 'bd_driver_twist.out']
# load_dir_ind = [0, 1, 5] # Flap, Edge, Twist

##### Stiffness Matrix from Dynamic Outputs

# Load K
with open('bd_driver.BD.sum.yaml') as f:
    data = list(yaml.load_all(f, Loader=SafeLoader))

dict_data = data[-1]
Kfull = np.array(dict_data['K_BD']) # This uses BD coordinate system, but the tip displacements may be in IEC. May be same here.
Kfull = Kfull[6:, 6:] # Apply fixed Boundary Conditions


##### Construct local stiffness based on linearization of angle conversions

applied_mag = 1000

local_flexibility = construct_flexibility(Kfull, applied_mag)

Klocal = np.linalg.inv(local_flexibility/applied_mag)

# Angle Conversion here
# d c / d phi (c = 0) = 1
# Therefore, don't actually need to convert the stiffness matrix.

print('Klocal:')
print(Klocal)

print('Norm of Klocal[:, 2]: {}'.format(np.linalg.norm(Klocal[:, 2])))

##### Stiffness Matrix for Local with angle conversion prior to inversion
applied_mags = [1, 100, 1000, 5000]

for ind in range(len(applied_mags)):

    local_flexibility = construct_flexibility(Kfull, applied_mags[ind])

    # print(local_flexibility / applied_mags[ind])

    # Angle Conversion here
    local_flexibility[:, 2] = 4 * np.arctan(local_flexibility[:, 2]/4.0)

    # Invert to get stiffness matrix
    Klocal_pre = np.linalg.inv(local_flexibility/applied_mags[ind])

    # Compare stiffness matrices
    print('Applied Load: {},   Error in Klocal[:, 2]: {}'.format(applied_mags[ind], 
            np.linalg.norm(Klocal[:, 2] - Klocal_pre[:, 2])))
