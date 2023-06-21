
import numpy as np
import matplotlib.pyplot as plt

def mass_norm_bend_modes(x, L, beta, rho):
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

def unif_modal_bend_loading(L, beta, rho):
    """
    Modal loading for a constant load distribution
    For bending modes
    """

    val = 2.0 / np.sqrt(rho*L) / beta * (np.sin(beta*L) - np.sinh(beta*L)) \
                 / (np.cos(beta*L) + np.cosh(beta*L))

    return val

def bending_freq(inertia, stiffness, L, betaL):
    """
    Calculate the bending frequency of the first bending mode for the structure
    """

    omega = betaL**2 / L**2 * np.sqrt(stiffness / inertia)

    return omega

def mass_norm_tors_modes(x, L, rotJ, rho):
    """
    Mass normalized mode shapes for the case of torsion
    """

    val = np.sqrt(2 / rotJ / rho / L) * np.sin(0.5*np.pi*x / L)

    return val 

def unif_modal_tors_loading(L, rotJ, rho):
    """
    Calculates the modal loading for the torsion mode
    """
    
    val = 2 / np.pi * np.sqrt(2* L / rotJ / rho)

    return val

def triangle_modal_tors_loading(L, rotJ, rho):
    """
    Calculates the modal loading for the torsion mode
    Applies a triangular load proportional to the spanwise position
    """
    
    val = 4 * L**2 / (np.pi**2) * np.sqrt(2 / L / rotJ / rho)

    return val

def torsion_freq(inertia, stiffness, L):
    """
    Calculate the torsion frequency of the first torsion mode
    """

    omega = 0.5 * np.pi/ L * np.sqrt(stiffness / inertia)

    # print('Torsional Inputs')
    # print('stiffness: {}'.format(stiffness))
    # print('inertia: {}'.format(inertia))

    return omega


def construct_analytical_model(L, beta, span_frac, stiffness_vec, inertia_vec, betaL):
    """
    Construct a 3 DOF model using the analytical beam modes etc.
    """

    bending_phi = mass_norm_bend_modes(span_frac*L, L, beta, rho)
    torsion_phi = mass_norm_tors_modes(span_frac*L, L, rotJ, rho)

    Phi = np.diag([bending_phi, bending_phi, torsion_phi])

    # Frequencies for the three modes
    omega_bend = bending_freq(inertia_vec[:2], stiffness_vec[:2], L, betaL)
    omega_tors = torsion_freq(inertia_vec[-1], stiffness_vec[-1], L)

    omega = np.hstack((omega_bend, [omega_tors]))

    # Stiffness Matrix
    Kmat = np.diag(omega**2 / np.array([bending_phi, bending_phi, torsion_phi])**2)

    Mmat = np.diag(1.0 / np.array([bending_phi, bending_phi, torsion_phi])**2)

    # Modal Loading (uniform)
    bending_f = unif_modal_bend_loading(L, beta, rho)
    torsion_f = unif_modal_tors_loading(L, rotJ, rho)

    Ttrans = np.diag([bending_f / bending_phi, bending_f/bending_phi, torsion_f/torsion_phi])

    return omega, Kmat, Mmat, Ttrans, Phi

if __name__=="__main__":

    ############
    # Parameters of the model
    
    # Beta * L for the first mode from pg 396 of Fundamentals of Structual Dynamics by Craig and Kurdila
    betaL = 1.875 # 1.8751 
    
    L = 1.0e2 # m
    

    span_frac = 0.7344240000000001 # - comes from different node for different element order on simple model. 0.7826179487179487
   
    density = 8e3 # kg/m^3

    Emod = 2e11 # Elastic Modulus, Pa
    # shearMod = 1.4e11 # Pa

    # Bulkmod = 1.40e11 # Pa, Bulk modulus
    poisson = 0.3
    shearMod = Emod / 2.0 / (1 + poisson)
    # shearMod = 3 * Bulkmod * Emod / (9 * Bulkmod - Emod)

    chord = 5.0 # chord length, m
    width = 1.5 # width of rectangle, m
    
    #########
    # Calculated Parameters
    
    beta = betaL / L

    span_pos = span_frac * L
   
    # Textbook:
    # torsion calculation
    a = chord / 2.0
    b = width / 2.0

    # a > b with each side being 2a or 2b
    Ktorsion = a*b**3*(16.0/3.0 - 3.36*b/a*(1 - b**4/12/a**4)) # Textbook


    # # Website with a and b being the full length and a < b - the two options match
    # a = width
    # b = chord
    # Ktorsion2 = a**3*b*(1/3.0 - 0.21*a/b*(1 - a**4 / 12.0 / b**4)) # https://structx.com/Shape_Formulas_024.html
    # print('Two Torsion Stiffnesses:')
    # print([Ktorsion,  Ktorsion2])
 
    # Stiffness for flap, edge, torsion directions
    # Bending directions should be E*I
    # Torsion direction should be G*Ktorsion
    stiffness_vec = np.array([Emod*chord*width**3/12, Emod*chord**3*width/12, Ktorsion*shearMod])

    rotJ = chord*width/12*(chord**2 + width**2) # rotational inertia per unit length for the torsion mode
    rho = density*chord*width # density per unit length

    inertia_vec = np.array([rho, rho, rotJ*rho])

    # # For Debugging Inputs
    # print('Stiffness Vec:')
    # print(stiffness_vec)

    # print('Inertia Vec:')
    # print(inertia_vec)

    # print('[E, G]')
    # print([Emod, shearMod])

    #########
    # Check Mode Shape Normalization 
    xplot = np.linspace(0, L, 200) 
    phi_norm = mass_norm_bend_modes(xplot, L, beta, rho)
    
    integral = np.trapz(phi_norm**2 * rho, x=xplot)
    print('Mass norm check of analytical bending modes (should be 1): {}'.format(integral))
    
    xplot = np.linspace(0, L, 200) 
    phi_norm = mass_norm_tors_modes(xplot, L, rotJ, rho)
    
    integral = np.trapz(phi_norm**2 * rotJ*rho, x=xplot)
    print('Mass norm check of analytical torsion modes (should be 1): {}'.format(integral))

    #########
    # Plot the mode shape to check
    xplot = np.linspace(0, L, 200)
    phi = mass_norm_bend_modes(xplot, L, beta, rho)
    
    phi_tors = mass_norm_tors_modes(xplot, L, rotJ, rho)

    plt.plot(xplot, phi, label='Bending')
    plt.plot(xplot, phi_tors, label='Torsion')
    plt.legend()
    plt.xlim((0, L)) 
    plt.savefig('FirstModes.png')

    #########
    # Generate the system

    omega, Kmat, Mmat, Ttrans, Phi = construct_analytical_model(L, beta, span_frac, stiffness_vec, inertia_vec, betaL)

    print('Frequencies: {} Hz'.format(str((omega/2/np.pi).tolist())))

    print('\nMass:')
    print(Mmat)

    print('\nStiffness:')
    print(Kmat)

    print('\nForce Transform Matrix:')
    print(Ttrans)

    print('\nMode Shape Matrix:')
    print(Phi)

    print('Could add a numerical check for the modal uniform loading values')

    print('\nUniform loading for bending modes:')

    xplot = np.linspace(0, L, 200) 
    phi_norm = mass_norm_bend_modes(xplot, L, beta, rho)

    bending_force_analytic = unif_modal_bend_loading(L, beta, rho)
    
    integral = np.trapz(phi_norm, x=xplot)
    print('Uniform Modal Loading Calculation bending: {}'.format(integral))
    print('Analytical modal loading bending: {}'.format(bending_force_analytic))

    print('\nUniform loading for torsion modes:')

    xplot = np.linspace(0, L, 200) 
    phi_norm = mass_norm_tors_modes(xplot, L, rotJ, rho)

    force_analytic = unif_modal_tors_loading(L, rotJ, rho)
    
    integral = np.trapz(phi_norm, x=xplot)
    print('Uniform Modal Loading Calculation: {}'.format(integral))
    print('Analytical modal loading: {}'.format(force_analytic))

    print('\n\nThese outputs miss some coordinate changes that are automatically for even 0 aoa for cfd v. beamdyn coordinates.')


    # Verify the twist force transformation entry for the triangular load
    triang_load = triangle_modal_tors_loading(L, rotJ, rho) / span_pos # with 1 at spanwise position
    phi_twist_n = mass_norm_tors_modes(span_frac*L, L, rotJ, rho)

    print('\nTwist Force Transform Matrix Entry (triangle load proportional to z): {}'.format(triang_load / phi_twist_n ))
    

