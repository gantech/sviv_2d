"""
Functions for running HHT Alpha algorithm
"""

import numpy as np


def hht_alpha_predict(xn, vn, an, xnm1, vnm1, anm1, dt):
    """ 
    Predict states to pass to aero model
    """

    xpred_np1 = xn + dt*(1.5*vn - 0.5*vnm1);   

    vpred_np1 = vn + dt*(1.5*an - 0.5*anm1);  

    return xpred_np1, vpred_np1


def hht_alpha_update(xn, vn, an, Mmat, Cmat, Kmat, alpha, tn, dt, Fn, Fnp1,
                     load_scale=1.0, T=np.eye(3)):
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

    Fext = Fnp1*(1 + alpha) - alpha * Fn

    Fext = T @ (load_scale * Fext)

    right_vec = -Cmat @ (vn + (1+alpha)*dt*(1-gamma)*an) \
                -Kmat @ (xn + (1+alpha)*dt*vn + (1+alpha)*(dt**2)/2*(1-2*beta)*an) \
                + Fext

    # Solve for time n+1 accel
    anp1 = np.linalg.solve(left_mat, right_vec)

    # Evaluate displacement and velocity at n+1
    xnp1 = xn + dt*vn + (dt**2)/2*((1 - 2*beta)*an + 2*beta*anp1)

    vnp1 = vn + dt*((1 - gamma)*an + gamma*anp1)

    return xnp1, vnp1, anp1


def hht_alpha_integrate(x0, v0, Mmat, Cmat, Kmat, alpha, dt, Fextfun, t0, t1, load_scale=1.0, T=np.eye(3)):
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
    right_vec_0 = Fextfun(t0, x0, v0) - Cmat @ v0 - Kmat @ x0

    a0 = np.linalg.solve(Mmat, right_vec_0)

    # Store initial states
    xhist[:, 0] = x0
    vhist[:, 0] = v0
    ahist[:, 0] = a0

    # Time n-1 states for predictor
    xnm1 = np.zeros_like(x0)
    vnm1 = np.zeros_like(x0)
    anm1 = np.zeros_like(x0)

    thist = t0 + dt*np.array(range(time_steps+1))


    # Run time integration
    for tind in range(time_steps):

        # Predict
        xpred_np1,vpred_np1 = hht_alpha_predict(xhist[:, tind], vhist[:, tind], ahist[:, tind], xnm1, vnm1, anm1, dt)

        # Evaluate Forces
        Fn = Fextfun(t0+dt*tind, xhist[:, tind], vhist[:, tind])
 
        Fnp1 = Fextfun(t0+dt*(tind+1), xpred_np1, vpred_np1)

        # Update step
        xnp1, vnp1, anp1 = hht_alpha_update(xhist[:, tind], vhist[:, tind], ahist[:, tind], 
                                           Mmat, Cmat, Kmat, alpha, 
                                           t0+dt*tind, dt, Fn, Fnp1,
                                           load_scale=load_scale, T=T)

        xhist[:, tind+1] = xnp1
        vhist[:, tind+1] = vnp1
        ahist[:, tind+1] = anp1

        xnm1 = xhist[:, tind]
        vnm1 = vhist[:, tind]
        anm1 = ahist[:, tind]

    return thist, xhist, vhist, ahist

