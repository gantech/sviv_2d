"""
Python solves of the unit tests for the Airfoil SMD
"""

import numpy as np
from scipy.linalg import eigh

import hht_alpha_fun as hhta

######
# Approximate matrices (rounded) from a 3 DOF model

M = np.array([7000.0, -90.0, 60.0, -90.0, 6400.0, 1800.0, 60.0, 1800.0, 7800]).reshape(3,3)

K = np.array([ 131000.0, 7700.0, -97000.0, 7700.0, 66000.0, -31000.0, -97000.0, -31000.0, 4800000.0]).reshape(3,3)
C = np.array([300.0, 8.0, -70.0, 8.0, 200.0, 20.0, -70.0, 20.0, 3700.0]).reshape(3,3)

T = np.array([62.0, 0.0, 10.0, 0.0, 61.0, 0.0,  1.0, -6.0, 60.0]).reshape(3,3)

Tident = np.eye(3)

######
# Verify rounded matrices still make sense

eigvals,eigvecs = eigh(K, M)

print(eigvals)

print(eigvecs.T @ C @ eigvecs)


#####
# Do an update

print('Starting update test.')

xn = np.array([0.1, 0.2, 0.01])
vn = np.array([0.0, 0.0, 0.0])
an = np.array([0.0, 0.0, 0.0])
Fn = np.array([0.0, 0.0, 0.0])
Fnp1 = np.array([25000.0, 35000.0, 50000.0])

load_scale = 1.0 # needs to be 1.0 for these tests since it is not in this function in nalu-wind

tn = 0.0
dt = 0.01
alpha = -0.05

xnp1, vnp1, anp1 = hhta.hht_alpha_update(xn, vn, an, M, C, K, alpha, tn, dt, 
                      Fn, Fnp1, load_scale=load_scale, T=Tident)

print('\nUpdate step with identity Tmap')
print('X, V, A')
print(str(xnp1.tolist()))
print(str(vnp1.tolist()))
print(str(anp1.tolist()))

#####
# Do an update with the force mapping

K2 = K / 100.0

xn = np.array([0.1, 0.2, 0.01])
vn = np.array([0.2, 0.3, -0.05]) # Add a nonzero value 
an = np.array([0.3, 0.2, 0.0]) # Add a nonzero value
Fn = np.array([200.0, 400.0, 50.0]) # Add a nonzero value
Fnp1 = np.array([250.0, 350.0, 500.0])

load_scale = 1.0 # 1 for the tests since not scaled here in naluwind.

tn = 0.0
dt = 0.05
alpha = -0.05

xnp1, vnp1, anp1 = hhta.hht_alpha_update(xn, vn, an, M, C, K2, alpha, tn, dt, 
                      Fn, Fnp1, load_scale=load_scale, T=T)

print('\nUpdate step with nonidentity Tmap')
print('X, V, A')
print(str(xnp1.tolist()))
print(str(vnp1.tolist()))
print(str(anp1.tolist()))

#####
# Do a predict step

xn = np.array([0.1, 0.2, 0.01])
vn = np.array([1.0, -2.0, 0.05])
an = np.array([-0.5, -7.0, 1.0])

xnm1 = np.array([0.0, 0.0, 0.0]) # doesn't matter for predictor
vnm1 = np.array([1.1, -1.4, -0.05])
anm1 = np.array([0.5, -8.0, 2.0])

dt = 0.01

xpred_np1, vpred_np1 = hhta.hht_alpha_predict(xn, vn, an, xnm1, vnm1, anm1, dt)


print('\nPredict step')
print('X, V')
print(str(xpred_np1.tolist()))
print(str(vpred_np1.tolist()))

#####
# Test advance supported by update steps
print('\nStarting Advance test')

K2 = K / 100.0

xn = np.array([0.1, 0.2, 0.01])
vn = np.array([0.2, 0.3, -0.05]) # Add a nonzero value 
an = np.array([0.3, 0.2, 0.0]) # Add a nonzero value
Fn = np.array([200.0, 400.0, 50.0]) # Add a nonzero value
Fnp1 = np.array([250.0, 350.0, 500.0])
Fnp2 = np.array([350.0, 450.0, 700.0])

xnp1, vnp1, anp1 = hhta.hht_alpha_update(xn, vn, an, M, C, K2, alpha, tn, dt, 
                      Fn, Fnp1, load_scale=load_scale, T=T)


xnp2, vnp2, anp2 = hhta.hht_alpha_update(xnp1, vnp1, anp1, M, C, K2, alpha, tn+dt, dt, 
                      Fnp1, Fnp2, load_scale=load_scale, T=T)

print('\nTwo update steps with nonidentity Tmap')
print('X1, V1, A1')
print(str(xnp1.tolist()))
print(str(vnp1.tolist()))
print(str(anp1.tolist()))

print('')
print('X2, V2, A2')
print(str(xnp2.tolist()))
print(str(vnp2.tolist()))
print(str(anp2.tolist()))
