
import numpy as np
import matplotlib.pyplot as plt

def mass_norm_modes(x, L, beta, rho):
    """
    Beam Modes that are mass normalized
    Cantilever beam with x=0 fixed

    Inputs:
      x - locations to evaluate
      L - length of beam
      beta - parameter of beam modes (depends on mode number)
      rho - mass per unit length of the beam
    """

    val = 1.0 / np.sqrt(rho * L) * ( np.cos(beta * x) - np.cosh(beta*x) \
            + (np.sin(beta*L) - np.sinh(beta*L)) / (np.cos(beta*L) + np.cosh(beta*L)) \
            * (np.sin(beta*x) - np.sinh(beta*x)) )

    return val

def unif_modal_loading(L, beta, rho):
    """
    Modal loading for a constant load distribution
    """

    val = -2.0 * (np.cosh(beta*L) - 1.0)*(np.sin(beta*L) - np.sinh(beta*L)) \
           / np.sqrt(rho*L) / beta / (np.cos(beta*L) - np.cosh(beta*L))

    return val 

def construct_analytical_model(L, beta, rho, rotJ, span_frac, stiffness_vec, betaL):
    """
    Construct a 3 DOF model using the analytical beam modes etc.
    """

    bending_phi = mass_norm_modes(span_frac*L, L, beta, rho)
    torsion_phi = mass_norm_modes(span_frac*L, L, beta, rotJ)

    Phi = np.diag([bending_phi, bending_phi, torsion_phi])

    # Inertia for the directions of the modes - per unit length
    inertia_vec = np.array([rho, rho, rotJ])

    # Frequencies for the three modes
    omega = betaL**2 / L**2 * np.sqrt(stiffness_vec / inertia_vec)

    # Stiffness Matrix
    Kmat = np.diag(omega**2 / np.array([bending_phi, bending_phi, torsion_phi])**2)

    Mmat = np.diag(1.0 / np.array([bending_phi, bending_phi, torsion_phi])**2)

    # Modal Loading (uniform)
    bending_f = unif_modal_loading(L, beta, rho)
    torsion_f = unif_modal_loading(L, beta, rotJ)

    Ttrans = np.diag([bending_f / bending_phi, bending_f/bending_phi, torsion_f/torsion_phi])

    return omega, Kmat, Mmat, Ttrans

if __name__=="__main__":

    ############
    # Parameters of the model
    
    # Beta * L for the first mode from pg 396 of Fundamentals of Structual Dynamics by Craig and Kurdila
    betaL = 1.8751 
    
    L = 2.0 # m
    

    span_frac = 0.7826179487179487
   
    density = 1.3 

    Emod = 1e6 # Elastic Modulus, Pa
    poisson = 0.3
    chord = 2.8 # chord length, m
    width = 1 # width of rectangle, m
    
    #########
    # Calculated Parameters
    
    beta = betaL / L

    span_pos = span_frac * L
   
    shearMod = Emod / 2.0 / (1 + poisson)

    # torsion calculation
    a = chord / 2.0
    b = width / 2.0

    Ktorsion = a*b**3*(16.0/3.0 - 3.36*b/a*(1 - b**4/12/a**4))
 
    # Stiffness for flap, edge, torsion directions
    # Bending directions should be E*I
    # Torsion direction should be G*J
    stiffness_vec = np.array([Emod*chord*width**3/12, Emod*chord**3*width/12, Ktorsion*shearMod])

    rotJ = density*chord*width/12*(chord**2 + width**2) # rotational inertia per unit length for the torsion mode
    rho = density*chord*width # density per unit length

    # # For Debugging Inputs
    # print('Stiffness Vec:')
    # print(stiffness_vec)

    # print('Inertia Vec:')
    # print([rho, rho, rotJ])

    # print('[E, G]')
    # print([Emod, shearMod])

    #########
    # Check Mode Shape Normalization 
    xplot = np.linspace(0, L, 200) 
    phi_norm = mass_norm_modes(xplot, L, beta, rho)
    
    integral = np.trapz(phi_norm**2 * rho, x=xplot)
    print('Mass norm check of analytical modes (should be 1): {}'.format(integral))
    
    #########
    # Plot the mode shape to check
    xplot = np.linspace(0, L, 200)
    phi = mass_norm_modes(xplot, L, beta, rho)
    
    plt.plot(xplot, phi)
    plt.xlim((0, L)) 
    plt.savefig('FirstMode.png')

    #########
    # Generate the system

    omega, Kmat, Mmat, Ttrans = construct_analytical_model(L, beta, rho, rotJ, span_frac, stiffness_vec, betaL)

    print('Frequencies: {} Hz'.format(str(omega.tolist())))

    print('\nMass:')
    print(Mmat)

    print('\nStiffness:')
    print(Kmat)

    print('\nForce Transform Matrix:')
    print(Ttrans)
