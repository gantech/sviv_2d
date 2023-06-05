# coding: utf-8
"""
Load Mass and Stiffness Matrices from yaml file to do modal analysis

"""

import numpy as np

import yaml
from yaml.loader import SafeLoader

from scipy.linalg import eigh

with open('../bd_driver.BD.sum.yaml') as f:
    data = list(yaml.load_all(f, Loader=SafeLoader))

dict_data = data[-1]

# print(dict_data.keys())

Mfull = np.array(dict_data['M_BD'])
Kfull = np.array(dict_data['K_BD'])

Mfull_IEC = np.array(dict_data['M_IEC'])
Kfull_IEC = np.array(dict_data['K_IEC'])

# Check symmetry:
print('norm(K) = {}'.format(np.linalg.norm(Kfull)))
print('norm(K-K^T) = {}'.format(np.linalg.norm(Kfull-Kfull.T)))
print('norm(K_BD - K_IEC) = {}'.format(np.linalg.norm(Kfull-Kfull_IEC)))

print('')

print('norm(M) = {}'.format(np.linalg.norm(Mfull)))
print('norm(M-M^T) = {}'.format(np.linalg.norm(Mfull-Mfull.T)))
print('norm(M_BD - M_IEC) = {}'.format(np.linalg.norm(Mfull-Mfull_IEC)))

# Mass and stiffness in a different coordinate system. 
# Mfull = np.array(dict_data['M_IEC'])
# Kfull = np.array(dict_data['K_IEC'])

# Apply Boundary Conditions
print('\nApplying Boundary conditions\n')
Mfull = Mfull[6:, 6:]
Kfull = Kfull[6:, 6:]

print('Mass shape: {}'.format(Mfull.shape))


subset_by_index = [0, 59]
eigvals, eigvecs = eigh(Kfull, Mfull, subset_by_index=subset_by_index)

# Check positive definiteness
# eigvals, eigvecs = eigh(Kfull, subset_by_index=subset_by_index) # Not positive definite
# eigvals, eigvecs = eigh(Mfull, subset_by_index=subset_by_index) # is positive definite

eigvals.sort()

print('w^2:')
print(eigvals)

print('f [Hz]')
print(np.sqrt(eigvals)/2.0/np.pi)

