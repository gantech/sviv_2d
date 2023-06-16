"""
Test the displacement for the 3 DOF model against analytically calculated displacements
"""

import numpy as np

import yaml
from yaml.loader import SafeLoader 

from scipy.linalg import eigh

############################
# User Inputs

# print('\nUniform Load Distribution:')
# yamlfile = 'SimpleBeam_3DOF_AOA5.yaml'

print('\nTriangle Load Distribution:')
yamlfile = 'SimpleBeam_3DOF_AOA5_triangle.yaml'

# print('\nRoot Triangle Load Distribution:')
# yamlfile = 'SimpleBeam_3DOF_AOA5_root_triangle.yaml'

aoa = 5.0 # deg

bend_load = 100 # N/m distributed load for bending
moment_load = 100 # Nm / m distributed moment for twist


############################
# Load YAML file

# Load YAML file
with open(yamlfile) as f:
    data = list(yaml.load_all(f, Loader=SafeLoader))

dict_data = data[-1]

# print(dict_data.keys())

Mmat = np.array(dict_data['mass_matrix']).reshape(3,3)
Kmat = np.array(dict_data['stiffness_matrix']).reshape(3,3)
Cmat = np.array(dict_data['damping_matrix']).reshape(3,3)

Tmat = np.array(dict_data['force_transform_matrix']).reshape(3,3)

############################
# Calculate displacements

flap_vec = np.array([np.sin(np.pi/180*aoa), np.cos(np.pi/180*aoa), 0]).reshape(-1,1)
edge_vec = np.array([np.cos(np.pi/180*aoa), -np.sin(np.pi/180*aoa), 0]).reshape(-1,1)
twist_vec = np.array([0,0,1]).reshape(-1,1)

loadcases = Tmat @ np.hstack((flap_vec*bend_load, edge_vec*bend_load, twist_vec*moment_load))

dispcases = np.linalg.solve(Kmat, loadcases)


############################
# Print Results

# print('Approx coords [Edge, Flap, Rot]')
# 
# print('\nEdge Load Displacements [m]:')
# print(dispcases[:, 0])
# 
# 
# print('\nFlap Load Displacements [m]:')
# print(dispcases[:, 1])
# 
# print('\nTwist Load Displacements [rad]:')
# print(dispcases[:, 2])

def print_disp(name, load_vec, disp_vec):
    """ 
    Quick function to output all of the displacement metrics to double check for 3 loads
    """

    print('\n' + name)
    
    print('Magnitude of Bending Displacement: {:e}'.format(np.linalg.norm(disp_vec[:-1])))
    print('Magnitude of Twist: {:e}'.format(disp_vec[-1]))


    dot_prod = (disp_vec[:-1] @ load_vec[:-1] / np.linalg.norm(disp_vec[:-1]) / np.linalg.norm(load_vec[:-1]))

    print('Norm Dot Product (1 is parallel) : {}'.format(dot_prod))


print_disp('Flap', flap_vec, dispcases[:, 0])
print_disp('Edge', edge_vec, dispcases[:, 1])
print_disp('Twist', twist_vec, dispcases[:, 2])


print('Note: These will never match analytical exactly because these are projected through the mode shapes')


############################
# Check Modal Analysis Problem of the 3 DOF

print('\n\nChecking Modal Properties')

expected_Phi = np.hstack((flap_vec, edge_vec, twist_vec))

# Solve eigenvalue problem
eigvals,eigvecs = eigh(Kmat, Mmat, [0,2])

print('Natural Frequencies:')
print(np.sqrt(eigvals)/2/np.pi)

# Mode Shapes on rotated coordinates
print('\nModal Orthogonality Check with Expected Modes - should be Identity')

mode_shape_prod = expected_Phi.T @ Mmat @ eigvecs

print(mode_shape_prod / ( np.diag(mode_shape_prod).reshape(1, -1)))


# Damping Factors
zeta_vec = (eigvecs.T @ Cmat @ eigvecs) / 2.0 / np.sqrt(eigvals)

print('\nDamping Factors [on diagonal, off should be 0]')
print(zeta_vec)

