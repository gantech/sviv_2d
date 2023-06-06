"""
Process nalu-wind outputs from a time integration for an unforced case. 

These can be compared to unforced vibration for verification. 
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

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


#########################
# Load the data set

df = nc.Dataset(output_nc)

t = np.array(df['time'][:])
x = np.array(df['x'][:])
xdot = np.array(df['xdot'][:])
forces = np.array(df['f'][:])

print('{} Timesteps Loaded.'.format(t.shape[0]))

#########################
# Plot displacement history for quick check

plt.plot(t, x[:, 0], label='x')
plt.plot(t, x[:, 1], label='y')
plt.plot(t, x[:, 2], label='Theta')

plt.xlim((t[0], t[-1]))

plt.legend()

plt.savefig('disp_hist.png')

plt.close()

#########################
# Load Parameters to verify

Mmat = np.array(df['mass_matrix'][:])
Kmat = np.array(df['stiffness_matrix'][:])
Cmat = np.array(df['damping_matrix'][:])

x0 = x[0, :]
v0 = xdot[0, :]

#########################
# Run the python HHT Alpha implementation to check against

t0 = 0.0
t1 = t[-1]

dt = t[1] - t[0]

Fextfun = lambda t,x,v : np.zeros_like(x)

thist, xhist, vhist, ahist = hht.hht_alpha_integrate(x0, v0, Mmat, Cmat, Kmat, alpha,
                                                     dt, Fextfun, t0, t1, load_scale=0.0)

#########################
# Plot displacement history for quick check

plt.plot(t, x[:, 0], label='nalu - x')
plt.plot(t, x[:, 1], label='nalu - y')
plt.plot(t, x[:, 2], label='nalu - Theta')

plt.plot(thist, xhist[0, :], '--', label='x')
plt.plot(thist, xhist[1, :], '--', label='y')
plt.plot(thist, xhist[2, :], '--', label='Theta')

plt.xlim((t[0], t[-1]))

plt.legend()

plt.savefig('disp_hist_compare.png')

plt.close()


#########################
# Analytically calculate the solution with modal analysis.

xhist_a = vutils.modal_time_series(Mmat, Kmat, Cmat, x0, thist)


rms_py_hht = np.sqrt(np.sum((xhist - xhist_a)**2, axis=1) / xhist_a.shape[1])

rms_nalu_hht = np.sqrt(np.sum((x.T - xhist_a)**2, axis=1) / xhist_a.shape[1])

rms_hht_diff = np.sqrt(np.sum((x.T - xhist)**2, axis=1) / xhist.shape[1])

print('RMS Error for Python HHT Alpha (to analytical): ' + str(rms_py_hht.tolist()))
print('RMS Error for NALU   HHT Alpha (to analytical): ' + str(rms_nalu_hht.tolist()))
print('RMS Diff for NALU v. python  HHT Alpha: ' + str(rms_hht_diff.tolist()))


plt.plot(t, x[:, 0], label='nalu - x')
plt.plot(t, x[:, 1], label='nalu - y')
plt.plot(t, x[:, 2], label='nalu - Theta')

plt.plot(thist, xhist_a[0, :], '--', label='x')
plt.plot(thist, xhist_a[1, :], '--', label='y')
plt.plot(thist, xhist_a[2, :], '--', label='Theta')

plt.xlim((t[0], t[-1]))

plt.legend()

plt.savefig('disp_hist_analytical.png')

plt.close()
