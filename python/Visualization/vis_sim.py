""" 
Script for visualizing data from CFD runs as an initial pass
"""

import numpy as np
import netCDF4 as nc


import yaml
from yaml.loader import SafeLoader 

import matplotlib.pyplot as plt

from scipy.linalg import eigh


#########################################
# User Inputs

filename = 'af_smd_deflloads.nc'

ignore_force_indices = 5 # number of forces to zero out at the beginning for sudden transients. 

yaml3dof = './chord_3dof.yaml'


#########################################
# Load the data file

data = nc.Dataset(filename)

time = np.array(data['time'][:])
x = np.array(data['x'][:])
xdot = np.array(data['xdot'][:])

forces = np.array(data['f'][:]) 

forces[:ignore_force_indices, :] = 0.0


with open(yaml3dof) as f:
    struct_data = list(yaml.load_all(f, Loader=SafeLoader))

Tforce = np.array(struct_data[0]['force_transform_matrix']).reshape(3,3)

forces = (Tforce @ forces.T).T

#########################################
# Convert to Modal Domain

Mmat = np.array(data['mass_matrix'][:])
Kmat = np.array(data['stiffness_matrix'][:])
Cmat = np.array(data['damping_matrix'][:])

eigvals,eigvecs = eigh(Kmat,Mmat, subset_by_index=[0, Mmat.shape[0]-1])

qhist = (eigvecs.T @ Mmat @ x.T).T
fqhist = (eigvecs.T @ Mmat @ forces.T).T

#########################################
# Plot Time Series of Displacements


plt.plot(time, x[:, 0], label='x')
plt.plot(time, x[:, 1], label='y')
plt.plot(time, x[:, 2]*180/np.pi, label='theta [deg]')

plt.xlim((time[0], time[-1]))
plt.ylabel('Displacement')

plt.legend()

plt.savefig('xhist.png')
plt.close()


#########################################
# Plot Time Series of Forces


fig, axs = plt.subplots(2)

axs[0].plot(time, forces[:, 0], label='Fx')
axs[0].plot(time, forces[:, 1], label='Fy')
axs[0].set_xlim((time[0], time[-1]))
axs[0].set_ylabel('Force')

axs[0].legend()

axs[1].plot(time, forces[:, 2], label='Moment')
axs[1].set_xlim((time[0], time[-1]))
axs[1].set_ylabel('Moment')


axs[1].legend()

fig.savefig('forcehist.png')
plt.close(fig)


#########################################
# Plot Modal Displacements

fig, axs = plt.subplots(2)

axs[0].plot(time, qhist[:, 0], label='Flap')
axs[0].plot(time, qhist[:, 1], label='Edge')
axs[0].set_xlim((time[0], time[-1]))
axs[0].set_ylabel('Modal Disp')

axs[0].legend()

axs[1].plot(time, qhist[:, 2], label='Torsion')
axs[1].set_xlim((time[0], time[-1]))
axs[1].set_ylabel('Model Disp')


axs[1].legend()

fig.savefig('qhist.png')
plt.close(fig)

#########################################
# Plot Modal Displacements

plt.plot(time, fqhist[:, 0], label='flap mode')
plt.plot(time, fqhist[:, 1], label='edge mode')
plt.plot(time, fqhist[:, 2], label='torsion mode')

plt.xlim((time[0], time[-1]))
plt.ylabel('Modal Force')

plt.legend()

plt.savefig('fqhist.png')
plt.close()

