"""
Script for verifying the axis that BeamDyn rotations are around
"""
import numpy as np

import sys
sys.path.append('..')
import construct_utils as cutils


### Off section distance (off axis section is displaced with x_kp equal to this)
x_kp_off = -100

### Matrices with top section on axis
# File with the stiffness matrix
on_K_file = 'bd_outputs/bd_dyn_on_axis.BD.sum.yaml'


### Matrices with top section off axis
# File with stiffness matrix
off_K_file = 'bd_outputs/bd_dyn_off_axis.BD.sum.yaml'

### Load Stiffness matrices
Mmat_on,  Kmat_on,  node_coords_on,  quad_coords_on  = cutils.load_M_K_nodes(on_K_file)
Mmat_off, Kmat_off, node_coords_off, quad_coords_off = cutils.load_M_K_nodes(off_K_file)

### Apply a moment at the mid point of the beam
node_force = 5 # python index of nodes [0, 10]

Fmoment = np.zeros(Mmat_on.shape[0])
Fmoment[node_force*6+5] = 1e5

### Apply Boundary Conditions
Kbc_on = Kmat_on[6:, 6:]
Kbc_off = Kmat_off[6:, 6:]
Fmoment = Fmoment[6:]


disp_on = np.linalg.solve(Kbc_on, Fmoment)
disp_off = np.linalg.solve(Kbc_off, Fmoment)

### Print Relevant Outputs:

print('\nRotations at the Tip')
print('on axis section: {}'.format(disp_on[-1]))
print('off axis section: {}'.format(disp_off[-1]))

print('\nX Translation at Tip')
print('on axis section: {}'.format(disp_on[-6]))
print('off axis section: {}'.format(disp_off[-6]))

print('\nY Translation at Tip')
print('on axis section: {}'.format(disp_on[-5]))
print('off axis section: {}'.format(disp_off[-5]))

print('\nExpected Translation for off-axis section: small angle Y')
print(disp_off[-1] * x_kp_off)
