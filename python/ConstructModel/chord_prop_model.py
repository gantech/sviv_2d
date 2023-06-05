"""
Script to construct a model with distributed loads applied.
Distributed loads are applied proportional to the chord of
the airfoil. 

Script creates a 2D, 3DOF model for a section of the IEA-15 MW

Frequencies match those of the full blade (calculated with BeamDyn matrices)

Damping factors are set at the top to match IEA-15 MW
"""


import numpy as np
import yaml

import sys
sys.path.append('..')
import construct_utils as cutils


###########################################
##### Define Input Parameters

# BeamDyn yaml file results from a dynamic analysis 
bd_yaml = '../bd_driver.BD.sum.yaml'

# BeamDyn node of interest for section used to construct the 3DOF model
#   Note that node 0 corresponds to the root and the fixed boundary condition.
node_interest = 7 # python index [0-10]

# Directions to apply loads to the static model
#   interested in flap, edge, and twist loads/motions
load_directions = [0, 1, 5]

# Python indices of modes of interest
#   interested in 1st flap, 1st edge, 1st twisting modes
mode_indices = [0, 1, 5]

# Modal Damping Factors for the modes of interest.
#   These are taken from:
#   https://github.com/IEAWindTask37/IEA-15-240-RWT/wiki/Frequently-Asked-Questions-(FAQ)#what-are-the-values-of-structural-damping
# This uses fraction of critical damping.
zeta_3dof = np.array([0.48e-2, 0.48e-2, 1e-2])

# Output file name
#   Output file will contain: mass, stiffness, damping matrices
out_3dof = 'chord_3dof.yaml'

# Mode Shape Importance Order
#   Local indices of which mode shapes are most important of the first 3
#   The more important mode shapes will be better preserved in the final 
#   3DOF model. The least important mode shapes will need to be altered
#   so that all modes are orthogonal (by definition)
mode_ortho_order = [0, 1, 2]

# Load Distribution Definition
#   load_span - defines the spanwise locations that distributed load is defined at 
#   load_val  - distributed load value at defined coordinates
#   Calculation assumes linear interpolation between defined locations for distributed load
#   For the present case, it is assumed equal for all three load directions
#   The present case uses proportional to the chord length of the IEA-15 MW turbine. 
#   Values are copied from here:
#       https://github.com/IEAWindTask37/IEA-15-240-RWT/blob/master/WT_Ontology/IEA-15-240-RWT.yaml

load_span = np.array([0.0, 0.02040816326530612, 0.04081632653061224, 0.061224489795918366, 0.08163265306122448, 0.1020408163265306, 0.12244897959183673, 0.14285714285714285, 0.16326530612244897, 0.18367346938775508, 0.2040816326530612, 0.22448979591836732, 0.24489795918367346, 0.26530612244897955, 0.2857142857142857, 0.3061224489795918, 0.32653061224489793, 0.3469387755102041, 0.36734693877551017, 0.3877551020408163, 0.4081632653061224, 0.42857142857142855, 0.44897959183673464, 0.4693877551020408, 0.4897959183673469, 0.5102040816326531, 0.5306122448979591, 0.5510204081632653, 0.5714285714285714, 0.5918367346938775, 0.6122448979591836, 0.6326530612244897, 0.6530612244897959, 0.673469387755102, 0.6938775510204082, 0.7142857142857142, 0.7346938775510203, 0.7551020408163265, 0.7755102040816326, 0.7959183673469387, 0.8163265306122448, 0.836734693877551, 0.8571428571428571, 0.8775510204081632, 0.8979591836734693, 0.9183673469387754, 0.9387755102040816, 0.9591836734693877, 0.9795918367346939, 0.985, 0.99, 0.995, 1.0])

load_val  = np.array([5.2, 5.208839941579524, 5.237887092263203, 5.293325313383697, 5.3673398548149205, 5.452092684226667, 5.5400317285038465, 5.621824261194381, 5.692531175149338, 5.74261089072697, 5.764836827022541, 5.756119529852528, 5.70309851275065, 5.604676021602162, 5.471559126660524, 5.322778014171772, 5.16648228816705, 5.019421327310202, 4.885807888739599, 4.767959675121795, 4.654566079625438, 4.54103105171191, 4.42817557762473, 4.316958876583997, 4.207880735790049, 4.101646187027423, 3.9987123353123564, 3.8994086760515647, 3.803172543681295, 3.7093894536544263, 3.6171117415725322, 3.525634918177657, 3.434082670567315, 3.341933111457596, 3.2486784477614132, 3.156109679927359, 3.0645800048338336, 2.9729926470824872, 2.8807051066906166, 2.786969376686517, 2.6910309386270574, 2.591965555977676, 2.4893236475052167, 2.383917231097341, 2.2759238162069977, 2.165466732053696, 2.0526250825584818, 1.9377533268191636, 1.819662967336163, 1.7799216728668075, 1.7077871948468315, 1.472482673397968, 0.5000000000000001])


# Angle of Attack Definition
# Initial Twist from BeamDyn at the spanwise location
#    Initial twist from beamdyn is applied to rotate from the IEC reference
#    coordinate system to align the system matrices with the airfoil coordinates
#    Here initial twist is linearly interpolated between two spanwise points 
#    from the BeamDyn input file for the IEA 15MW
#    The initial twist is about the negative z-axis
#
# NOTE: The spanwise position that twist is calculated here is done 
#  independently of the node number selected. They should be done consistently.
span_loc = 9.15663E+01 
initial_twist = np.interp(span_loc, [9.07347e+01, 9.31224e+01],
                           [-1.62330e+00, -1.88437e+00 ])

print('The initial twist of the blade section is')
print(initial_twist)

# Angle of Attack of the airfoil (relative to chord)
angle_attack = 50 # Degrees
print('Angle of attack is:')
print(angle_attack)


# print('Setting rotations to zero for initial check')
# angle_attack = 0
# initial_twist = 0

###########################################
##### Load Mass and Stiffness from BeamDyn

Mmat, Kmat, node_coords, quad_coords = cutils.load_M_K_nodes(bd_yaml)

# Apply boundary conditions
Mbc = Mmat[6:, 6:]
Kbc = Kmat[6:, 6:]

###########################################
##### Calculate the Load Distribution Function

# Size of Ndofs x 3
# Columns correspond to the 3 directions of loading

Psi = cutils.calc_load_dis(node_coords, quad_coords, load_span, load_val,
                           node_interest, load_directions, refine = False)

###########################################
##### Extract Desired Modal Properties from the global model

subset_eigvals, Phi_L, Phi_G = cutils.extract_local_subset_modes(Kbc, Mbc, 
                                         mode_indices, node_interest,
                                         load_directions)

print('Initial Extracted Phi_L')
print(Phi_L)

Lambda = np.diag(subset_eigvals)

###########################################
##### Construct the 3 DOF to match global properties

Phi_L_inv = np.linalg.inv(Phi_L)

# Construct matrices
Mlocal = Phi_L_inv.T @ Phi_L_inv
Klocal = Phi_L_inv.T @ Lambda @ Phi_L_inv
Tcfd_local = Phi_L_inv.T @ (Phi_G.T @ Psi)

# Construct damping matrix
modal_damp = np.sqrt(Lambda) * np.diag(2*zeta_3dof)
Clocal = Phi_L_inv.T @ modal_damp @ Phi_L_inv


print('\nOrtho Checks with initially constructed matrices and Phi_L')

print('\nOrtho Checks:')
print('Mass:')
print(Phi_L.T @ Mlocal @ Phi_L)

print('Stiffness:')
print(Phi_L.T @ Klocal @ Phi_L)

print('Damping:')
print(Phi_L.T @ Clocal @ Phi_L)


###########################################
# Repeat Eigenanalysis for verification
eigvals_3dof,eigvecs_3dof = cutils.modal_analysis(Klocal, Mlocal) 


###########################################
##### Transform to the CFD coordinate system

# Coordinate transform
M3dof, K3dof, C3dof, T3dof = cutils.transform_mats(Mlocal, Klocal, Clocal, Tcfd_local,
                                         initial_twist, angle_attack)

# Redo Eigenanalysis for the output checks
eigvals_3dof,eigvecs_3dof = cutils.modal_analysis(K3dof, M3dof) 

###########################################
##### Print results for sanity check

print('\nGlobal Modal Properties')
print('Frequencies [Hz]:')
print(np.sqrt(subset_eigvals)/2/np.pi)

print('Mode Shapes at DOFs (renormalized to Mlocal):')
norm_vals = np.sqrt(np.diag(Phi_L.T @ Mlocal @ Phi_L))
subset_phi_renorm = Phi_L / (np.ones((3,1)) * norm_vals.reshape(1,-1) )
# print(Phi_L) # Not normalized to the mass matrix, so have to rescale to compare.
print(subset_phi_renorm)


print('\nKlocal:')
print(Klocal)

print('\nLocal Modal Properties')
print('Frequencies [Hz]:')
print(np.sqrt(eigvals_3dof)/2/np.pi)

print('Mode Shapes at DOFs:')
print(eigvecs_3dof)

print('\nOrtho Checks:')
print('Mass:')
print(eigvecs_3dof.T @ M3dof @ eigvecs_3dof)

print('Stiffness:')
print(eigvecs_3dof.T @ K3dof @ eigvecs_3dof)

print('Damping:')
print(eigvecs_3dof.T @ C3dof @ eigvecs_3dof)

print('Frequencies from Klocal Check')
omega_klocal = np.sqrt(np.diag(eigvecs_3dof.T @ K3dof @ eigvecs_3dof))
print(omega_klocal / 2 /np.pi)

print('Damping factors from Clocal Check')
zeta_clocal = np.diag(eigvecs_3dof.T @ C3dof @ eigvecs_3dof)/omega_klocal/2
print(zeta_clocal)

print('\nForce Transform Matrix:')
print(T3dof)

###########################################
##### Save results to a file for nalu-wind inputs

print('\nWriting Mass, Stiffness, and Damping Matrices to:')
print(out_3dof)


with open(out_3dof, 'w') as f:
    f.write('mass_matrix : ' + str(M3dof.reshape(-1).tolist()) + '\n')
    f.write('stiffness_matrix : ' + str(K3dof.reshape(-1).tolist()) + '\n')
    f.write('damping_matrix : ' + str(C3dof.reshape(-1).tolist()) + '\n')
    f.write('force_transform_matrix : ' + str(T3dof.reshape(-1).tolist()) + '\n')

print('Finished writing output')


