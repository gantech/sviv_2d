"""
Run HHT Alpha in python and check the convergence rate w.r.t. time step

The script post_proc_unforced.py verifies that the python and nalu-wind implementations
match for the unforced case. (alpha=0.0)

This shows close to convergence proportional to dt^2, but doesn't seem to be too exact
"""

import netCDF4 as nc
import numpy as np

import sys
sys.path.append('../../HHTAlpha')
import hht_alpha_fun as hht

sys.path.append('..')
import verify_utils as vutils

#########################
# Input Parameters

# File with the nalu-wind resulst
output_nc = 'af_smd_deflloads.nc'

# time integration parameter alpha used for nalu-wind.
alpha = 0.0

dt_baseline = 0.01 /4
dt_refine   = 0.005/4


t0 = 0.0
t1 = 5.0

#########################
# Load the data set - get system properties

df = nc.Dataset(output_nc)

Mmat = np.array(df['mass_matrix'][:])
Kmat = np.array(df['stiffness_matrix'][:])
Cmat = np.array(df['damping_matrix'][:])

x = np.array(df['x'][:])
xdot = np.array(df['xdot'][:])

x0 = x[0, :]
v0 = xdot[0, :]*0.0 # analytical can't do initial velocity yet. 


#########################
# Run the python HHT Alpha implementation to check against

Fextfun = lambda t,x,v : np.zeros_like(x)

thist, xhist, vhist, ahist = hht.hht_alpha_integrate(x0, v0, Mmat, Cmat, Kmat, alpha,
                                                     dt_baseline, Fextfun, t0, t1, load_scale=0.0, a0_pred=True)

thist_refine, xhist_refine, vhist_refine, ahist_refine = hht.hht_alpha_integrate(x0, v0, Mmat, Cmat, Kmat, alpha,
                                                     dt_refine, Fextfun, t0, t1, load_scale=0.0, a0_pred=True)

#########################
# Analytically calculate the solution with modal analysis.

xhist_a,vhist_a = vutils.modal_time_series(Mmat, Kmat, Cmat, x0, thist_refine)

#########################
# Error Comparisons

print('error baseline: ' + str((xhist[:, -1] - xhist_a[:, -1]).tolist()))
print('error refined: ' + str((xhist_refine[:, -1] - xhist_a[:, -1]).tolist()))


# Energy Calculations
E_base = 0.5*(xhist[:, -1] @ Kmat @ xhist[:, -1]) + 0.5*(vhist[:, -1] @ Mmat @ vhist[:, -1])

E_refine = 0.5*(xhist_refine[:, -1] @ Kmat @ xhist_refine[:, -1]) + 0.5*(vhist_refine[:, -1] @ Mmat @ vhist_refine[:, -1])


E_a = 0.5*(xhist_a[:, -1] @ Kmat @ xhist_a[:, -1]) + 0.5*(vhist_a[:, -1] @ Mmat @ vhist_a[:, -1])

print('Energies/error:')
print('Analytical: {}'.format(E_a))
print('Baseline error: {}'.format(E_base - E_a))
print('Refined error:  {}'.format(E_refine - E_a))
