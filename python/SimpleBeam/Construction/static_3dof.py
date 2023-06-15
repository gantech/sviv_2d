"""
Test the displacement for the 3 DOF model against analytically calculated displacements
"""

import numpy as np

import yaml
from yaml.loader import SafeLoader 

############################
# User Inputs


yamlfile = 'SimpleBeam_3DOF.yaml'

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

Tmat = np.array(dict_data['force_transform_matrix']).reshape(3,3)

############################
# Calculate displacements

loadcases = Tmat @ np.diag(np.array([bend_load, bend_load, moment_load]))

dispcases = np.linalg.solve(Kmat, loadcases)


############################
# Print Results

print(['Edge, Flap, Rot'])

print('\nEdge Load Displacements [m]:')
print(dispcases[:, 0])


print('\nFlap Load Displacements [m]:')
print(dispcases[:, 1])

print('\nTwist Load Displacements [rad]:')
print(dispcases[:, 2])

print('Note: These will never match analytical exactly because these are projected through the mode shapes')

