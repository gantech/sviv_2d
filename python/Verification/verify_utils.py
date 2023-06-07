"""
Functions for verifying nalu wind outputs
"""

import numpy as np
from scipy.linalg import eigh


def modal_time_series(Mmat, Kmat, Cmat, x0, thist):
    """
    Calculate a time series response with modal analysis for initial displacement
    Only considers unforced.

    Inputs:
      Mmat - Mass matrix
      Kmat - Stiffness matrix
      Cmat - Damping matrix
      x0 - initial displacements
      thist - time history to evaluate at

    Outputs:
      xhist - displacement history

    NOTES:
      1. Assumes only initial displacement and zero initial velocity
      2. Assumes diagonal modal damping. May be incorrect unless Cmat is appropriately constructed.
    """

    sub_inds = [0, Mmat.shape[0]-1]

    eigvals, eigvecs = eigh(Kmat, Mmat, subset_by_index=sub_inds)
    
    # Spectral Matrix
    Lambda = np.diag(eigvals)
    ModalDamping = eigvecs.T @ Cmat @ eigvecs
    
    # frequencies
    omega = np.sqrt(eigvals)
    zeta = np.diag(ModalDamping)/2/omega
    
    # damped frequency
    omega_d = omega * np.sqrt(1 - zeta**2)

    # Initial Modal Displacements
    modal_q0 = eigvecs.T @ Mmat @ x0

    qhist = np.zeros((x0.shape[0], thist.shape[0]))
    qdothist = np.zeros((x0.shape[0], thist.shape[0]))

    # Evaluate sdof response
    for i in range(x0.shape[0]):
        qhist[i, :] += np.exp(-zeta[i]*omega[i]*thist)*modal_q0[i]*np.cos(omega_d[i]*thist)

        qdothist[i, :] += np.exp(-zeta[i]*omega[i]*thist)*modal_q0[i] \
                           *(-np.cos(omega_d[i]*thist)*omega[i]*zeta[i] \
                             -omega_d[i]*np.sin(omega_d[i]*thist) )
    
    xhist = eigvecs @ qhist
    vhist = eigvecs @ qdothist

    return xhist,vhist




