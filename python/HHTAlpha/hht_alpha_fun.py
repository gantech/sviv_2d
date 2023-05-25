"""
Functions for running HHT Alpha algorithm
"""

import numpy as np


def hht_alpha_update(xn, vn, an, Mmat, Cmat, Kmat, alpha, tn, dt, Fextfun):
    """
    Advance one step with the HHT-Alpha method
    
    Inputs:
        xn - displacement state at time n
        vn - velocity state at time n
        an - acceleration state at time n
        Mmat - Mass matrix
        Cmat - Damping Matrix
        Kmat - Stiffness matrix
        alpha - control parameter for HHT alpha algorithm, [-1/3, 0]
        tn - time n value
        dt - time step
        Fextfun - external force function of time
    
    Outputs:
        xnp1 - displacement state at time n+1
        vnp1 - velocity state at time n+1
        anp1 - acceleration state at time n+1
    
    """

    # Additional parameters based on alpha
    gamma = (1 - 2*alpha)/2
    beta = (1 - alpha)**2 / 4

    # Algorithm
    # Initial step formulate:
    #     left_mat @ anp1 = right_vec
    
    left_mat = Mmat + ((1+alpha)*dt*gamma)*Cmat + ((1+alpha)*(dt**2)*beta)*Kmat

    Fext = Fextfun(tn + (1+alpha)*dt)

    right_vec = -Cmat @ (vn + (1+alpha)*dt*(1-gamma)*an) \
                -Kmat @ (xn + (1+alpha)*dt*vn + (1+alpha)*(dt**2)/2*(1-2*beta)*an) \
                + Fext

    # Solve for time n+1 accel
    anp1 = np.linalg.solve(left_mat, right_vec)

    # Evaluate displacement and velocity at n+1
    xnp1 = xn + dt*vn + (dt**2)/2*((1 - 2*beta)*an + 2*beta*anp1)

    vnp1 = vn + dt*((1 - gamma)*an + gamma*anp1)

    return xnp1, vnp1, anp1


def hht_alpha_integrate(x0, v0, Mmat, Cmat, Kmat, alpha, dt, Fextfun, t0, t1):
    """
    Run HHT-Alpha integration problem
    
    Inputs:
        x0 - displacement state at time t0
        v0 - velocity state at time t0
        Mmat - Mass matrix
        Cmat - Damping Matrix
        Kmat - Stiffness matrix
        alpha - control parameter for HHT alpha algorithm, [-1/3, 0]
        dt - time step
        Fextfun - external force function of time
        t0 - initial time
        t1 - final time
    
    Outputs:
        thist - list of times for the history
        xhist - displacement state at times
        vhist - velocity state at times
        ahist - acceleration state at times
    
    """
    
    # Initialize Memory
    time_steps = int(np.ceil((t1 - t0)/dt))

    Ndof = Mmat.shape[0] # Number of degrees of freedom

    xhist = np.zeros((Ndof, time_steps+1))
    vhist = np.zeros((Ndof, time_steps+1))
    ahist = np.zeros((Ndof, time_steps+1))

    # Calculate Initial Acceleration
    #   M a + C v + K x = Fext
    right_vec_0 = Fextfun(t0) - Cmat @ v0 - Kmat @ x0

    a0 = np.linalg.solve(Mmat, right_vec_0)

    # Store initial states
    xhist[:, 0] = x0
    vhist[:, 0] = v0
    ahist[:, 0] = a0

    thist = t0 + dt*np.array(range(time_steps+1))


    # Run time integration
    for tind in range(time_steps):

        xnp1, vnp1, anp1 = hht_alpha_update(xhist[:, tind], vhist[:, tind], ahist[:, tind], \
                                            Mmat, Cmat, Kmat, \
                                            alpha, t0+dt*tind, dt, Fextfun)
        
        xhist[:, tind+1] = xnp1
        vhist[:, tind+1] = vnp1
        ahist[:, tind+1] = anp1

    return thist, xhist, vhist, ahist

