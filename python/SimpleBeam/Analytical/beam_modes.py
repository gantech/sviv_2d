
import numpy as np
import matplotlib.pyplot as plt

def unscaled_modes(x, L, beta):
    """
    Mode shapes without mass normalization
    """

    
    val = (np.sin(beta*L) - np.sinh(beta*L))*(np.sin(beta*x) - np.sinh(beta*x)) \
          + (np.cos(beta*L) - np.cosh(beta*L))*(np.cos(beta*x) - np.cosh(beta*x)) 

    return val

def mass_norm_val(L, beta, rho):
    """
    Calculate the mass normalization constant for the mode shape
    """

    # 1 = A^2*rho * inv_Asqrho for mass normalization
    inv_Asqrho = (0.25/beta) * ( np.sin(beta*L) * np.sinh(beta*L) )**2 * ( -np.sin(2*beta*L) + np.sinh(2*beta*L) \
         + 4*np.cos(beta*L)*np.sinh(beta*L) - 4*np.sin(beta*L)*np.cosh(beta*L) ) \
         + (1.0/beta) * (np.sin(beta*L) - np.sinh(beta*L) )**3 * (np.cos(beta*L) - np.cosh(beta*L)) \
         + (0.25/beta) * (np.cos(beta*L)*np.cosh(beta*L))**2 * (4.0*beta*L + np.sin(2*beta*L) \
         + np.sinh(2*beta*L) - 4.0*np.cos(beta*L)*np.sinh(beta*L) - 4.0*np.sin(beta*L)*np.cosh(beta*L) )

    print(inv_Asqrho)

    Asq = 1.0 / inv_Asqrho / rho

    return np.sqrt(Asq)

def mass_norm_modes(x, L, beta, rho):

     unscaled_val = unscaled_modes(x, L, beta)
     
     norm_val = mass_norm_val(L, beta, rho)

     return unscaled_val / norm_val


##################
# Actual Calls to function etc

# Beta * L for the first mode from pg 396 of Fundamentals of Structual Dynamics by Craig and Kurdila
betaL = 1.8751 

L = 1 # m

rho = 1.0 # density per unit length

#### Calculated Parameters

beta = betaL / L

#### Check Mode Shape Normalization

xplot = np.linspace(0, L, 200)

phi_norm = mass_norm_modes(xplot, L, beta, rho)

integral = np.trapz(phi_norm**2 * rho, xplot)

print(integral)

####
# Plot the mode shape to check

xplot = np.linspace(0, L, 200)
# phi = unscaled_modes(xplot, L, beta)
phi = mass_norm_modes(xplot, L, beta, rho)

plt.plot(xplot, phi)
plt.xlim((0, L))

plt.savefig('FirstMode.png')

