"""
Process nalu-wind outputs from a time integration for an unforced case. 

These can be compared to unforced vibration for verification. 
"""

import numpy as np
import netCDF4 as nc

import yaml
from yaml.loader import SafeLoader 

import matplotlib.pyplot as plt

import sys
sys.path.append('../../HHTAlpha')
import hht_alpha_fun as hht

sys.path.append('..')
import verify_utils as vutils



#########################
# Lookup Aero Forces from a table given conditions

def aero_lookup(ref_dict, x, v, initial_aoa_deg, chord, uinf, rho, extrudeLength=1):
    """
    Function to lookup aerodynamic loads given reference dictionary

    Inputs:
      ref_dict - dictionary of data relating AOA to lift/drag/moment
      x - vector of 3 positions [x, y, theta in radians around z+]
      v - vector of 3 velocities
      initial_aoa_deg - initial angle of attack in degrees
      chord - chord length of simulation. 
      uinf - free stream velocity [x+ direction]
      rho - density of fluid

    Outputs:
      loads - vector of aerodynamic loads [x, y, theta] directions
    """

    aoa = -x[-1]*180/np.pi + initial_aoa_deg

    cl = np.interp(aoa, ref_dict['aoa'], ref_dict['cl'])
    cd = np.interp(aoa, ref_dict['aoa'], ref_dict['cd'])
    cm = np.interp(aoa, ref_dict['aoa'], ref_dict['cm'])

    # Aero forces ignoring velocity of airfoil
    lift   = 0.5 * rho * uinf * uinf * cl * chord
    drag   = 0.5 * rho * uinf * uinf * cd * chord
    moment = -0.5 * rho * uinf * uinf * cm * chord * chord

    loads = np.array([drag, lift, moment])

    return loads

    
#########################
# Input Parameters

# File with the nalu-wind resulst
output_nc = 'af_smd_deflloads.nc'

# time integration parameter alpha used for nalu-wind.
alpha = 0.0

# Reference data for low angles of attack
ref_file = 'FFA-W3-211_rey05000000.yaml'
model_3dof_yaml = 'IEA15_aoa5_3dof.yaml'

uinf = 70.0 # m/s
rho = 1.225 # kg/m^3

show_nalu = False

extrude_multiple = 4.0 # number of chord lengths multiplied later.

t0 = 0.0
t1 = 20*20

reynolds = 5.0e6 


# These parameters cannot be set manually. They need to be updated in the 3DOF model, so are read from that output.
# initial_aoa_deg = 5 # deg
# chord = 2.8480588747143942 # m



#########################
# Load the data set for nalu-wind run

df = nc.Dataset(output_nc)

t = np.array(df['time'][:])
x = np.array(df['x'][:])
xdot = np.array(df['xdot'][:])
forces = np.array(df['f'][:])

print('{} Timesteps Loaded.'.format(t.shape[0]))

#########################
# Load data set for the lift/drag/moment calculations

with open(ref_file) as f:
    ref_data = list(yaml.load_all(f, Loader=SafeLoader))

ref_dict = ref_data[0]['FFA-W3-211']

#########################
# Load data set for the lift/drag/moment calculations

with open(model_3dof_yaml) as f:
    model_data = list(yaml.load_all(f, Loader=SafeLoader))

model_dict = model_data[0]


Mmat = np.array(model_dict['mass_matrix']).reshape(3,3)
Kmat = np.array(model_dict['stiffness_matrix']).reshape(3,3)
Cmat = np.array(model_dict['damping_matrix']).reshape(3,3)

force_transform = np.array(model_dict['force_transform_matrix']).reshape(3,3)

load_scale = model_dict['loads_scale'] 

chord = model_dict['chord_length']

extrude_length = extrude_multiple*chord

initial_aoa_deg = model_dict['angle']
print('Initial Angle of Attack: ' + str(initial_aoa_deg) + ' [deg]')

#########################
# Plot displacement history for quick check

if show_nalu:
    plt.plot(t, x[:, 0], label='x')
    plt.plot(t, x[:, 1], label='y')
    plt.plot(t, x[:, 2]*180/np.pi, label='Theta [deg]')
    
    plt.xlim((t[0], t[-1]))
    
    plt.legend()
    
    plt.savefig('disp_hist.png')
    
    plt.close()

#########################
# Load Parameters to verify

x0 = x[0, :] * 0.0
v0 = xdot[0, :] * 0.0

#########################
# Run the python HHT Alpha implementation to check against

dt = t[1] - t[0]

Fextfun = lambda t,x,v : aero_lookup(ref_dict, x, v, initial_aoa_deg, chord, uinf, rho, extrudeLength=extrude_length)

thist, xhist, vhist, ahist = hht.hht_alpha_integrate(x0, v0, Mmat, Cmat, Kmat, alpha,
                                                     dt, Fextfun, t0, t1, load_scale=load_scale, T=force_transform)

#########################
# Plot displacement history for quick check

if show_nalu:
    plt.plot(t, x[:, 0], label='nalu - x')
    plt.plot(t, x[:, 1], label='nalu - y')
    plt.plot(t, x[:, 2]*180/np.pi, label='nalu - Theta [deg]')

plt.plot(thist, xhist[0, :], '--', label='x - py')
plt.plot(thist, xhist[1, :], '--', label='y - py')
plt.plot(thist, xhist[2, :]*180/np.pi, '--', label='Theta [deg] - py')

plt.xlim((t0, t1))

plt.legend()

plt.savefig('disp_hist_compare.png')

plt.close()


# rms_hht_diff = np.sqrt(np.sum((x.T - xhist)**2, axis=1) / xhist.shape[1])
# 
# print('RMS Diff for NALU v. python  HHT Alpha: ' + str(rms_hht_diff.tolist()))

#########################
# Work out what the nalu-wind inputs should be to match re

viscosity = rho * uinf * chord / reynolds

print('Input Viscosity should be : {}'.format(viscosity))
