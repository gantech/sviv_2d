# coding: utf-8
import numpy as np

from hht_alpha_fun import hht_alpha_integrate

import sys
sys.path.append('..')
import construct_utils as cutils

##### Load Previously Constructed 3 DOF Matrices

mats_yaml = 'chord_3dof.yaml'

M3dof, C3dof, K3dof, T3dof = cutils.load3dof(mats_yaml)

##### Construct Problem Inputs

load_scale = 1.0

alpha = -0.05

x0 = np.zeros(M3dof.shape[0])
v0 = np.zeros(M3dof.shape[0])

t0 = 0
t1 = 5
dt = 0.01

freq = 1.0 # Hz
Famp = np.array([10e3, 0.0, 0.0]) # N

# Fextfun = lambda t : 5*t/t1*Famp*np.cos(freq*2*np.pi*t)
Fextfun = lambda t,x,v : Famp*np.cos(freq*2*np.pi*t)

##### Perform time integration

thist, xhist, vhist, ahist = hht_alpha_integrate(x0, v0, M3dof, C3dof, K3dof, alpha, dt, Fextfun, 
                                                 t0, t1, load_scale=load_scale, T=T3dof)

##### Visualize Results
import matplotlib.pyplot as plt

dof = 0

plt.plot(thist, xhist[dof, :], label='x')
plt.plot(thist, vhist[dof, :], label='v')
plt.plot(thist, ahist[dof, :], label='a')

plt.xlim((t0, t1))
plt.legend()

plt.savefig('./timehist.png')

