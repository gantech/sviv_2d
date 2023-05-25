"""
Construct a 3 DOF model matching the first flap/edge/twist frequencies

Utilize the linear mass and stiffness from BeamDyn. 

"""

import numpy as np

import yaml
from yaml.loader import SafeLoader

from scipy.linalg import eigh

##### Loading M and K

with open('bd_driver.BD.sum.yaml') as f:
    data = list(yaml.load_all(f, Loader=SafeLoader))

dict_data = data[-1]

# print(dict_data.keys())

Mfull = np.array(dict_data['M_BD'])
Kfull = np.array(dict_data['K_BD'])

# Apply Boundary Conditions
Mfull = Mfull[6:, 6:]
Kfull = Kfull[6:, 6:]

##### Modal Analysis

load_dir_ind = [0, 1, 5] # Flap, Edge, Twist
mode_inds = [0, 1, 5] # Python indices of modes - Flap, Edge, Twist

pos_ind = 5 # Output Node/Quad Point of interest (numbered starting from 1 after eliminating B.C.s)

subset_by_index = [0, 59]
eigvals, eigvecs = eigh(Kfull, Mfull, subset_by_index=subset_by_index)

# print('Eigenvector size and indices used for local eigenvectors:')
# print(eigvecs.shape)
# print((pos_ind-1)*6+load_dir_ind[0])
# print((pos_ind-1)*6+load_dir_ind[1])
# print((pos_ind-1)*6+load_dir_ind[2])

##### Reduced Mode Shapes
local_phi = np.zeros((3,3))

for mind in range(len(mode_inds)):
    for dind in range(len(load_dir_ind)):

        local_phi[dind, mind] = eigvecs[(pos_ind-1)*6+load_dir_ind[dind], mode_inds[mind]]

print(local_phi)

##### Stiffness Matrix for Local
applied_mag = 1000

local_flexibility = np.zeros((3,3))

for dind in range(len(load_dir_ind)):

    F = np.zeros(Mfull.shape[0])

    F[(pos_ind-1)*6+load_dir_ind[dind]] = applied_mag

    U_static = np.linalg.solve(Kfull, F)

    for rind in range(len(load_dir_ind)):

        local_flexibility[rind, dind] = U_static[(pos_ind-1)*6+load_dir_ind[rind]]

print('Local Flexibility')
print(local_flexibility)


print('\nLocal Stiffness')

Klocal = np.linalg.inv(local_flexibility/applied_mag)
print(Klocal)


print('\nLocal Phi^T K Phi')
print(local_phi.T @ Klocal @ local_phi)

##### Construct Mass Matrix

# Normalize the local mode shapes
local_eigs = np.array([eigvals[mode_inds[0]], eigvals[mode_inds[1]], eigvals[mode_inds[2]]])

local_scale = np.diag(local_phi.T @ Klocal @ local_phi)

# print(local_phi / local_phi[0:1, :])

# Rescaled Local Mode Shapes
local_phi = local_phi * np.sqrt(local_eigs/local_scale)

# print(local_phi / local_phi[0:1, :])

print('Local Mode Shapes Pre Mass')
print(local_phi)

print('Global Freq')
print(np.sqrt(local_eigs)/2/np.pi)

##### Apply Gram-Schmidt Orthogonalization to local mode shapes

def gram_schmidt(Klocal, local_phi, local_eigs):
    """
    Update the local eigenvectors to be orthogonal w.r.t. the local stiffness matrix
    In addition, phi^T K phi = diag(local_eigs) for normalizing new vectors
    """
    
    ortho_phi = np.copy(local_phi)

    for i in range(1,3):
        for j in range(i):

            proj = (ortho_phi[:, i:i+1].T @ Klocal @ ortho_phi[:, j:j+1])\
                    / (ortho_phi[:, j:j+1].T @ Klocal @ ortho_phi[:, j:j+1])\
                    * ortho_phi[:, j:j+1]

            ortho_phi[:, i:i+1] -= proj

    # Rescale modes to have correct normalization
    local_scale = np.diag(ortho_phi.T @ Klocal @ ortho_phi)
    ortho_phi = ortho_phi * np.sqrt(local_eigs/local_scale)

    return ortho_phi

# local_phi = gram_schmidt(Klocal, local_phi, local_eigs)
# 
# print('\nLocal Phi after applying orthogonalization:')
# print(local_phi)


##### Generate Mass Matrix and Recheck Eigenvalues / vectors

Mlocal = np.linalg.inv(local_phi).T @ np.linalg.inv(local_phi)

subset_by_index = [0, 2]
eigvals_3dof, eigvecs_3dof = eigh(Klocal, Mlocal, subset_by_index=subset_by_index)

print('\nLocal Frequencies')

print(np.sqrt(eigvals_3dof)/2/np.pi)

print('Freq. Error: %')

f_glob = np.sqrt(local_eigs)/2/np.pi
f_3dof = np.sqrt(eigvals_3dof)/2/np.pi
print((f_glob - f_3dof)/f_glob*100)


print('3 DOF Mode Shapes')
print(eigvecs_3dof)

print('Mass Ortho Check:')
print(eigvecs_3dof.T @ Mlocal @ eigvecs_3dof)

print('Mass Matrix')
print(Mlocal)


print('Stiffness Ortho Check:')
print(eigvecs_3dof.T @ Klocal @ eigvecs_3dof)

print('\nCheck MAC Values - rows=3DOF, cols=Global ')

MAC_vals = np.zeros((3,3))

for i in range(3):
    for j in range(3):
        MAC_vals[i, j] = (eigvecs_3dof[:, i] @ local_phi[:, j])**2 \
            / (eigvecs_3dof[:, i] @ eigvecs_3dof[:, i]) \
            / (local_phi[:, j] @ local_phi[:, j])

print(MAC_vals)


# Check Positive Definite Mass Matrix
subset_by_index = [0, 2]
eigvalsM, eigvecsM = eigh(Mlocal, subset_by_index=subset_by_index)

print('Check positive eigenvalues of Mlocal for positive definiteness')
print(eigvalsM)

##### Construct Damping Matrix

print('\nConstruct Damping Matrix:')

omega_3dof = np.sqrt(eigvals_3dof)

# For modal damping factors see:
#   https://github.com/IEAWindTask37/IEA-15-240-RWT/wiki/Frequently-Asked-Questions-(FAQ)#what-are-the-values-of-structural-damping
zeta_3dof = np.array([0.48e-2, 0.48e-2, 1e-2])

modal_damp = np.diag(2*omega_3dof*zeta_3dof)

# print(omega_3dof)
# print(modal_damp)

# Construct mode shape inverse matrix
phi_inv_3dof = eigvecs_3dof.T @ Mlocal

# Construct Clocal (physical coordinates damping matrix)
Clocal = phi_inv_3dof.T @ modal_damp @ phi_inv_3dof

print(Clocal)

print('\nClocal ortho check:')
print(eigvecs_3dof.T @ Clocal @ eigvecs_3dof)


##### Save the Local Matrices
np.savez('./local_mats.npz', M3dof=Mlocal, K3dof=Klocal, C3dof=Clocal)

