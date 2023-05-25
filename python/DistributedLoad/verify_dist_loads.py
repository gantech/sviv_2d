"""
Script for comparing the application of distributed loads
between BeamDyn with a constant load v. manually integrating
the constant load distribution and applying to BeamDyn stiffness
matrix
"""

import numpy as np

from pyFAST.input_output import FASTOutputFile
import pandas as pd


import sys
sys.path.append('..')
import construct_utils as cutils

import os

##### Define Comparison Cases

case_names = ['Flap', 'Edge', 'Twist']
static_bd = ['bd_driver_flap.out', 'bd_driver_edge.out', 'bd_driver_twist.out']
load_dir_ind = [0, 1, 5] # Flap, Edge, Twist

bd_dir = './beam_dyn_res'

fmag = 100 # magnitude of force/moment used in BeamDyn inputs

##### Get System Info From BeamDyn

Mmat,Kmat,node_coords,quad_coords = cutils.load_M_K_nodes('./beam_dyn_res/bd_driver.BD.sum.yaml')

# Apply boundary conditions to the stiffness matrix
K_bc = Kmat[6:, 6:]

##### Construct Integration Matrix Based on Trapezoid Rule

x_traps, int_mat_trap = cutils.construct_trap_int_mat(node_coords,
                                            quad_coords, refine=False)

##### Loop over load directions and compare

for ind in range(len(load_dir_ind)):

    print('\nComparing case: {}'.format(case_names[ind]))

    ### Manually Calculate Displacements

    # quadrature forces
    quad_force = fmag*np.ones(int_mat_trap.shape[1])

    dir_force = int_mat_trap @ quad_force

    # Define direction that load should be applied in
    dof_vec = np.zeros(6)
    dof_vec[load_dir_ind[ind]] = 1.0

    tot_force_vec = np.kron(dir_force, dof_vec) 

    force_bc = tot_force_vec[6:]

    # manual displacement results
    disp_man = np.linalg.solve(K_bc, force_bc)
    man_tip = disp_man[-6:]

    ### Load BeamDyn Results

    # BeamDyn Data Frame of outputs
    bd_df = FASTOutputFile(os.path.join(bd_dir, static_bd[ind])).toDataFrame()

    bd_np = bd_df[['TipTDxr_[m]', 'TipTDyr_[m]', 'TipTDzr_[m]', 'TipRDxr_[-]', 'TipRDyr_[-]', 'TipRDzr_[-]']].to_numpy()
    bd_tip = bd_np[-1]

    ### Compare BeamDyn / Manual Calculation

    print('Fraction Error by components:')
    print(np.abs( (man_tip - bd_tip)/bd_tip))

    print('Norm Error / norm BD:')
    print(np.linalg.norm(man_tip - bd_tip) / np.linalg.norm(bd_tip))
