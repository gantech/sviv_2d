import numpy as np
from numpy.polynomial import polynomial as p
from numpy.polynomial import Polynomial as Poly

import yaml
from yaml.loader import SafeLoader 

import sys
sys.path.append('..')
import construct_utils as cutils


##### Get Node Positions from BeamDyn

Mmat,Kmat,node_coords,quad_coords = cutils.load_M_K_nodes('bd_driver.BD.sum.yaml')


nodes = node_coords[:, 2]

domain = np.array([nodes[0], nodes[-1]])

print('Nodes Coordinates real:')
print(nodes)

print('Nodes Coordinates Normalized:')
print(nodes / nodes[-1])

##### Construct shape functions for the nodes over the domain

polys = cutils.construct_shape_funs(nodes)

print('Finished Construction')

# print(polys[0])
# print(polys[1])
# print(polys[2])

##### Verify Integrals of Polynomials over domain
# 
# for i in range(nodes.shape[0]):
# 
#     polyint = polys[i].integ()
# 
#     print(polyint(domain[1]) - polyint(domain[0]))
# 

##### Construct An Integration Matrix

print('\nConstructing an Integration Matrix of Nodal force values to interpolate and integrate to nodal forces:') 

int_mat = np.zeros((nodes.shape[0], nodes.shape[0]))

for i in range(nodes.shape[0]):
    for j in range(nodes.shape[0]):

        polyprod = polys[i] * polys[j]

        # print(polyprod)

        polyintegrated = polyprod.integ()

        int_mat[i, j] = polyintegrated(domain[1]) \
                         -polyintegrated(domain[0])

print(int_mat)

nodal_vals = np.ones(int_mat.shape[0])

print(int_mat @ nodal_vals)

print('Total Applied Force for unit load: {}'.format(np.sum(int_mat @ nodal_vals)))

##### Verify Displacements for a uniform distributed load against BeamDyn

print('\nVerifying displacements for a uniform distributed load')

Kfull = np.copy(Kmat)

nodal_dis_force_vals = 100*np.ones(int_mat.shape[0])

nodal_forces = int_mat @ nodal_dis_force_vals

# convert nodal forces to be applied to all DOFs at each node instead of just 1 dof
dof_vec = np.zeros(6)
dof_vec[0] = 1.0
forces_full = np.kron(nodal_forces, dof_vec)


# apply boundary conditions to K and nodal forces
K_bc = Kfull[6:, 6:]
F_bc = forces_full[6:]

disp = np.linalg.solve(K_bc, F_bc)

print('Nodal Displacements for distributed load:')
print(disp)

##### Construct Integration Matrix Based on Trapezoid Rule

print('\n\nTrapezoid Integration:')

x_traps, int_mat_trap = cutils.construct_trap_int_mat(node_coords,
                                            quad_coords, refine=True)

# x_traps, w_traps = cutils.calc_trap_weights(quad_coords, refine=True)
# 
# print(quad_coords[:, 2].shape)
# print(x_traps.shape)
# print(w_traps.shape)
# 
# quad_span_coords = x_traps
# 
# int_mat_trap = np.zeros((nodes.shape[0], w_traps.shape[0]))
# 
# for i in range(nodes.shape[0]):
#     
#     shape_fun = polys[i](quad_span_coords)
# 
#     int_mat_trap[i, :] = shape_fun * w_traps
# 
# # print(int_mat_trap)

quad_vals = np.ones(int_mat_trap.shape[1])

print(int_mat_trap @ quad_vals)

print('Total Applied Force for unit load: {}'.format(np.sum(int_mat_trap @ quad_vals)))

quad_dis_force_vals = 100*np.ones(int_mat_trap.shape[1])

nodal_forces = int_mat_trap @ quad_dis_force_vals

# convert nodal forces to be applied to all DOFs at each node instead of just 1 dof
forces_full = np.kron(nodal_forces, dof_vec)


# apply boundary conditions to K and nodal forces
F_bc = forces_full[6:]

disp = np.linalg.solve(K_bc, F_bc)

print('Nodal Displacements for distributed load:')
print(disp)


