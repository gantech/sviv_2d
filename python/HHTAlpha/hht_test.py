# coding: utf-8
import numpy as np

from hht_alpha_fun import hht_alpha_integrate

##### Load Previously Constructed 3 DOF Matrices

mats_3dof = np.load('./local_mats.npz')

M3dof = mats_3dof['M3dof']
C3dof = mats_3dof['C3dof']
K3dof = mats_3dof['K3dof']

##### Construct Problem Inputs

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

thist, xhist, vhist, ahist = hht_alpha_integrate(x0, v0, M3dof, C3dof, K3dof, alpha, dt, Fextfun, t0, t1)

##### Visualize Results
import matplotlib.pyplot as plt

dof = 0

plt.plot(thist, xhist[dof, :], label='x')
plt.plot(thist, vhist[dof, :], label='v')
plt.plot(thist, ahist[dof, :], label='a')

plt.xlim((t0, t1))
plt.legend()

plt.savefig('./timehist.png')


### 2nd order convergence is not exactly achieved right now. Likely because the start of the integration is not exact.
# ##### Verify 2nd order convergence
# 
# t0 = 0.0
# t1 = 5
# dt = 0.01
# 
# fine_fac = 200
# 
# # Solution at base time step
# thist, xhist, vhist, ahist = hht_alpha_integrate(x0, v0, M3dof, C3dof, K3dof, alpha, dt, Fextfun, t0, t1)
# 
# # Half Time Step
# thist2, xhist2, vhist2, ahist2 = hht_alpha_integrate(x0, v0, M3dof, C3dof, K3dof, alpha, 0.5*dt, Fextfun, t0, t1)
# 
# # Reference Solution
# thist_ref, xhist_ref, vhist_ref, ahist_ref = hht_alpha_integrate(x0, v0, M3dof, C3dof, K3dof, alpha, dt/fine_fac, Fextfun, t0, t1)
# 
# error_base = np.linalg.norm(xhist - xhist_ref[:, ::fine_fac]) / thist.shape[0]
# error_half = np.linalg.norm(xhist2 - xhist_ref[:, ::int(fine_fac/2)]) / thist2.shape[0]
# 
# print('Error Base: {}, Error Half: {}, Reduction: {}'.format(error_base, error_half, error_base/error_half))
# 
# 
