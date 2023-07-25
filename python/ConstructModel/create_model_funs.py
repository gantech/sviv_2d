"""
File with function to create the full 3 DOF model for automation of test cases
"""

import numpy as np
import yaml

import os
my_dir=os.path.dirname(__file__)

import sys
sys.path.append(os.path.join(my_dir, '..'))
import construct_utils as cutils

def construct_3dof(bd_yaml, node_interest, out_3dof, angle_attack,
                   load_grid, load_val, 
                   twist_grid, twist_val,
                   pitch_axis_grid, pitch_axis_val,
                   chord_grid, chord_val,
                   load_directions=[0,1,5],
                   mode_indices=[0,1,5],
                   zeta_3dof = np.array([0.48e-2, 0.48e-2, 1e-2]),
                   extrude_length = 4):
    """
    Function for constructing 3dof model based on a blade

    Inputs:
      bd_yaml - BeamDyn summary file from a dynamic analysis to get M, K matrices
      node_interest - node of interest 0 is the root. Indices [0,10] for 10th order elem
      out_3dof - filename for the yaml output of the 3 DOF model
      angle_attack - angle of attack for the final blade and model (transform to CFD coords) [deg]
      load_grid - grid of points for load distribution. Loads in one direction get scaled
                   across the blade, but only for loads in that direction. Fraction of span.
      load_val - values of load distribution at grid
      twist_grid - grid of points for the initial twist angle for rotating the airfoil. Fraction span.
      twist_val - value of initial twist angles on grid. Taken from IEA Ontology, so in radians. 
      pitch_axis_grid - grid of points for the pitch axis. Fraction of span.
      pitch_axis_val - value of pitch axis at grid points
      chord_grid - grid to define the chord on. Fraction of span.
      chord_val - chord values on the grid
      load_directions - Directions that the airfoil is moving/being loaded in [flap, edge, torsion]
      mode_indices - modes that represent primary motions of interest 1st of flap,edge,torsion indices
      zeta_3dof - damping factors for the three modes of interest. Taken from IEA 15 MW for default. 
                   Fraction of critical damping.
      extrude_length - The length that the airfoil is extruded perpendicular to the section for CFD simulations. Number of chord lengths

    Output: 
      None - Results saved in out_3dof yaml file.
    """ 

    ########
    # Load BeamDyn file

    Mmat, Kmat, node_coords, quad_coords = cutils.load_M_K_nodes(bd_yaml)
    
    # Apply boundary conditions
    Mbc = Mmat[6:, 6:]
    Kbc = Kmat[6:, 6:]
    
    ########
    # Calculate the spanwise position for the node of interest + distributed loads
    
    Psi,span_loc,span_frac = cutils.calc_load_dis(node_coords, quad_coords, 
                                                  load_grid*node_coords[-1, 2], load_val,
                                                  node_interest, load_directions,
                                                  refine = False)
    
    ########
    # Interpolate values to span location

    initial_twist = np.interp(span_frac, twist_grid, twist_val, left=np.nan, right=np.nan)

    # convert initial twist to degrees
    initial_twist = initial_twist * 180 / np.pi

    pitch_axis = np.interp(span_frac, pitch_axis_grid, pitch_axis_val, left=np.nan, right=np.nan)

    chord_length = np.interp(span_frac, chord_grid, chord_val, left=np.nan, right=np.nan)
    # print(chord_length)

    # Dimensional airfoil with depth of 4 * chord_length
    loads_scale = 1.0 / (extrude_length * chord_length)

    ########
    # Extract Desired Modal Properties from Global

    subset_eigvals, Phi_L, Phi_G = cutils.extract_local_subset_modes(Kbc, Mbc, 
                                         mode_indices, node_interest,
                                         load_directions)

    Lambda = np.diag(subset_eigvals)


    ########
    # Construct the 3 DOF to match global properties
    
    Phi_L_inv = np.linalg.inv(Phi_L)
    
    # Construct matrices
    Mlocal = Phi_L_inv.T @ Phi_L_inv
    Klocal = Phi_L_inv.T @ Lambda @ Phi_L_inv
    Tcfd_local = Phi_L_inv.T @ (Phi_G.T @ Psi)
    
    # Construct damping matrix
    modal_damp = np.sqrt(Lambda) * np.diag(2*zeta_3dof)
    Clocal = Phi_L_inv.T @ modal_damp @ Phi_L_inv

    ########
    # Repeat Eigenanalysis for verification
    eigvals_3dof,eigvecs_3dof = cutils.modal_analysis(Klocal, Mlocal) 

    # match signs with previous eigenvectors
    eigvecs_3dof = eigvecs_3dof * np.sign(eigvecs_3dof[:1, :] * Phi_L[:1, :])


    tol_eigs = 1e-9

    if (np.max(np.abs(eigvals_3dof - subset_eigvals)) > tol_eigs) \
        or (np.max(np.abs(eigvecs_3dof-Phi_L)) > tol_eigs):

        print('Eigenvalues or eigenvectors have changed so exiting without completing.')
        print('Eigenvalue diff {}'.format(np.max(np.abs(eigvals_3dof - subset_eigvals))) )
        print('Eigenvec diff {}'.format(np.max(np.abs(eigvecs_3dof-Phi_L))) )

        sys.exit()

    # else:
    #     print('Eigenvalue diff {}'.format(np.max(np.abs(eigvals_3dof - subset_eigvals))) )
    #     print('Eigenvec diff {}'.format(np.max(np.abs(eigvecs_3dof-Phi_L))) )

    ########
    # Transform to the CFD coordinate system
    
    # Coordinate transform
    M3dof, K3dof, C3dof, T3dof = cutils.transform_mats(Mlocal, Klocal, Clocal, Tcfd_local,
                                             initial_twist, angle_attack)
    
    # Redo Eigenanalysis for the output checks
    eigvals_3dof,eigvecs_3dof = cutils.modal_analysis(K3dof, M3dof) 
    
    # Redo checks on eigenvalues
    if (np.max(np.abs(eigvals_3dof - subset_eigvals)) > tol_eigs):

        print('Eigenvalues have changed so exiting without completing.')
        print('Eigenvalue diff {}'.format(np.max(np.abs(eigvals_3dof - subset_eigvals))) )
        sys.exit()

    ########
    # Calculate ratios of primary motions between Phi_L and Phi_G
    Phi_L_tip = np.vstack((Phi_G[-6, :], Phi_G[-5, :], Phi_G[-1, :]))

    tip_ratio = np.diag(Phi_L_tip) / np.diag(Phi_L)

    ########
    # Write out the yaml file for the updated structure. 

    with open(out_3dof, 'w') as f:
        f.write('# mesh_transformation: \n')
        f.write('# - name: move_to_pitch_axis - (this value is non-dimensional, may need to be multipled by chord for dimensional simulations). \n')
        f.write('displacement: [{}, 0.0, 0.0] \n\n'.format(-pitch_axis))
    
        f.write('# - name: angle_of_attack \n')
        f.write('angle: {:.4f} \n\n'.format(angle_attack))
    
        f.write('# - type: airfoil_smd\n')
        f.write('mass_matrix : ' + str(M3dof.reshape(-1).tolist()) + '\n')
        f.write('stiffness_matrix : ' + str(K3dof.reshape(-1).tolist()) + '\n')
        f.write('damping_matrix : ' + str(C3dof.reshape(-1).tolist()) + '\n')
        f.write('force_transform_matrix : ' + str(T3dof.reshape(-1).tolist()) + '\n')
        f.write('loads_scale: {} \n\n'.format(loads_scale))
    
        f.write('# Misc Values for preparing runs.\n')
        f.write('chord_length: {}\n'.format(chord_length))
        f.write('span_fraction: {}\n'.format(span_frac))
        f.write('phi_matrix : ' + str(eigvecs_3dof.reshape(-1).tolist()) + '# mode shape matrix for reference \n')
        f.write('tip_per_cross : ' + str(tip_ratio.tolist()) + '# for reference tip displacement/cross section for flap,edge,twist for those modes respectively. \n')

    return

def construct_IEA15MW_chord(bd_yaml, out_3dof, angle_attack, node_interest=7):
    """
    Construct a 3 DOF model of the IEA 15 MW turbine. 
    
    This function defines inputs to construct_3dof for the specific turbine. 
    See that function for input documentation. 
    """

    # chord calculation grid.
    chord_grid = [0.0, 0.02040816326530612, 0.04081632653061224, 0.061224489795918366, 0.08163265306122448, 0.1020408163265306, 0.12244897959183673, 0.14285714285714285, 0.16326530612244897, 0.18367346938775508, 0.2040816326530612, 0.22448979591836732, 0.24489795918367346, 0.26530612244897955, 0.2857142857142857, 0.3061224489795918, 0.32653061224489793, 0.3469387755102041, 0.36734693877551017, 0.3877551020408163, 0.4081632653061224, 0.42857142857142855, 0.44897959183673464, 0.4693877551020408, 0.4897959183673469, 0.5102040816326531, 0.5306122448979591, 0.5510204081632653, 0.5714285714285714, 0.5918367346938775, 0.6122448979591836, 0.6326530612244897, 0.6530612244897959, 0.673469387755102, 0.6938775510204082, 0.7142857142857142, 0.7346938775510203, 0.7551020408163265, 0.7755102040816326, 0.7959183673469387, 0.8163265306122448, 0.836734693877551, 0.8571428571428571, 0.8775510204081632, 0.8979591836734693, 0.9183673469387754, 0.9387755102040816, 0.9591836734693877, 0.9795918367346939, 0.985, 0.99, 0.995, 1.0]    
    chord_val  = [5.2, 5.208839941579524, 5.237887092263203, 5.293325313383697, 5.3673398548149205, 5.452092684226667, 5.5400317285038465, 5.621824261194381, 5.692531175149338, 5.74261089072697, 5.764836827022541, 5.756119529852528, 5.70309851275065, 5.604676021602162, 5.471559126660524, 5.322778014171772, 5.16648228816705, 5.019421327310202, 4.885807888739599, 4.767959675121795, 4.654566079625438, 4.54103105171191, 4.42817557762473, 4.316958876583997, 4.207880735790049, 4.101646187027423, 3.9987123353123564, 3.8994086760515647, 3.803172543681295, 3.7093894536544263, 3.6171117415725322, 3.525634918177657, 3.434082670567315, 3.341933111457596, 3.2486784477614132, 3.156109679927359, 3.0645800048338336, 2.9729926470824872, 2.8807051066906166, 2.786969376686517, 2.6910309386270574, 2.591965555977676, 2.4893236475052167, 2.383917231097341, 2.2759238162069977, 2.165466732053696, 2.0526250825584818, 1.9377533268191636, 1.819662967336163, 1.7799216728668075, 1.7077871948468315, 1.472482673397968, 0.5000000000000001]

    # Load distribution grid    
    load_grid = np.array(chord_grid)
    load_val  = np.array(chord_val)

    # Pitch axis offset grid
    pitch_axis_grid = [0.0, 0.02040816326530612, 0.04081632653061224, 0.061224489795918366, 0.08163265306122448, 0.1020408163265306, 0.12244897959183673, 0.14285714285714285, 0.16326530612244897, 0.18367346938775508, 0.2040816326530612, 0.22448979591836732, 0.24489795918367346, 0.26530612244897955, 0.2857142857142857, 0.3061224489795918, 0.32653061224489793, 0.3469387755102041, 0.36734693877551017, 0.3877551020408163, 0.4081632653061224, 0.42857142857142855, 0.44897959183673464, 0.4693877551020408, 0.4897959183673469, 0.5102040816326531, 0.5306122448979591, 0.5510204081632653, 0.5714285714285714, 0.5918367346938775, 0.6122448979591836, 0.6326530612244897, 0.6530612244897959, 0.673469387755102, 0.6938775510204082, 0.7142857142857142, 0.7346938775510203, 0.7551020408163265, 0.7755102040816326, 0.7959183673469387, 0.8163265306122448, 0.836734693877551, 0.8571428571428571, 0.8775510204081632, 0.8979591836734693, 0.9183673469387754, 0.9387755102040816, 0.9591836734693877, 0.9795918367346939, 1.0]     
    pitch_axis_val  = [0.5045454545454545, 0.4900186808012221, 0.47270018284548393, 0.4540147730610375, 0.434647782591965, 0.4156278851950606, 0.3979378721273935, 0.38129960745617403, 0.3654920515699109, 0.35160780834472827, 0.34008443128769117, 0.3310670675965599, 0.3241031342163746, 0.3188472934612394, 0.3146895762675238, 0.311488897995355, 0.3088429219529899, 0.3066054031112312, 0.3043613335231313, 0.3018756624023877, 0.2992017656131912, 0.29648581499532917, 0.29397119399704474, 0.2918571873240831, 0.2901098902886204, 0.28880659979944606, 0.28802634398115073, 0.28784151044623507, 0.28794253614539367, 0.28852264941156663, 0.28957685074559625, 0.2911108045758606, 0.2930139151081327, 0.2952412111444283, 0.2977841397364215, 0.300565286724993, 0.3035753776130124, 0.30670446458784534, 0.30988253764299156, 0.3130107259708016, 0.31639042766652853, 0.32021109189825026, 0.32462311714967124, 0.329454188784972, 0.33463306413024474, 0.3401190402144396, 0.3460555975714659, 0.3527211856428439, 0.3600890296396286, 0.36818181818181805]

    #  twist_grid = [0.0, 0.02040816326530612, 0.04081632653061224, 0.061224489795918366, 0.08163265306122448, 0.1020408163265306, 0.12244897959183673, 0.14285714285714285, 0.16326530612244897, 0.18367346938775508, 0.2040816326530612, 0.22448979591836732, 0.24489795918367346, 0.26530612244897955, 0.2857142857142857, 0.3061224489795918, 0.32653061224489793, 0.3469387755102041, 0.36734693877551017, 0.3877551020408163, 0.4081632653061224, 0.42857142857142855, 0.44897959183673464, 0.4693877551020408, 0.4897959183673469, 0.5102040816326531, 0.5306122448979591, 0.5510204081632653, 0.5714285714285714, 0.5918367346938775, 0.6122448979591836, 0.6326530612244897, 0.6530612244897959, 0.673469387755102, 0.6938775510204082, 0.7142857142857142, 0.7346938775510203, 0.7551020408163265, 0.7755102040816326, 0.7959183673469387, 0.8163265306122448, 0.836734693877551, 0.8571428571428571, 0.8775510204081632, 0.8979591836734693, 0.9183673469387754, 0.9387755102040816, 0.9591836734693877, 0.9795918367346939, 1.0]
    # twist_val  = [0.27217629557079365, 0.27205736171561723, 0.26896980562529643, 0.26090151424382807, 0.24885698262522313, 0.23382423192614568, 0.21680513961749598, 0.1988727863224039, 0.18100767932828987, 0.16413076067279694, 0.1492522184680202, 0.1367143113540973, 0.12551282194274888, 0.11434749848233088, 0.10356776280240737, 0.0933068658568242, 0.0837118194006672, 0.07498970428955204, 0.06714203432398387, 0.06013256274855414, 0.053702278525950935, 0.047709665190988265, 0.04210143739978706, 0.03685595675838511, 0.03191153817565269, 0.027205903019809997, 0.022731811675977428, 0.018577762273585947, 0.014720912664231815, 0.011110140960669581, 0.00762759323174886, 0.00418280011853364, 0.0006925062977488528, -0.003016419809918337, -0.0071049531892497, -0.011874829529214339, -0.017440963696104015, -0.023047513564523475, -0.02833190814159476, -0.03288842847284285, -0.03641158408109318, -0.037768893627003246, -0.03797559234861602, -0.03761714725398, -0.036702695029715245, -0.035227785335510846, -0.033104414148684154, -0.030095161935387575, -0.026321748778324498, -0.021683756060763528]

    twist_grid = np.array([0.0, 0.02040816326530612, 0.04081632653061224, 0.061224489795918366, 0.08163265306122448, 0.1020408163265306, 0.12244897959183673, 0.14285714285714285, 0.16326530612244897, 0.18367346938775508, 0.2040816326530612, 0.22448979591836732, 0.24489795918367346, 0.26530612244897955, 0.2857142857142857, 0.3061224489795918, 0.32653061224489793, 0.3469387755102041, 0.36734693877551017, 0.3877551020408163, 0.4081632653061224, 0.42857142857142855, 0.44897959183673464, 0.4693877551020408, 0.4897959183673469, 0.5102040816326531, 0.5306122448979591, 0.5510204081632653, 0.5714285714285714, 0.5918367346938775, 0.6122448979591836, 0.6326530612244897, 0.6530612244897959, 0.673469387755102, 0.6938775510204082, 0.7142857142857142, 0.7346938775510203, 0.7551020408163265, 0.7755102040816326, 0.7959183673469387, 0.8163265306122448, 0.836734693877551, 0.8571428571428571, 0.8775510204081632, 0.8979591836734693, 0.9183673469387754, 0.9387755102040816, 0.9591836734693877, 0.9795918367346939, 1.0])
               
    twist_val = np.array( [0.27217629557079365, 0.27205736171561723, 0.26896980562529643, 0.26090151424382807, 0.24885698262522313, 0.23382423192614568, 0.21680513961749598, 0.1988727863224039, 0.18100767932828987, 0.16413076067279694, 0.1492522184680202, 0.1367143113540973, 0.12551282194274888, 0.11434749848233088, 0.10356776280240737, 0.0933068658568242, 0.0837118194006672, 0.07498970428955204, 0.06714203432398387, 0.06013256274855414, 0.053702278525950935, 0.047709665190988265, 0.04210143739978706, 0.03685595675838511, 0.03191153817565269, 0.027205903019809997, 0.022731811675977428, 0.018577762273585947, 0.014720912664231815, 0.011110140960669581, 0.00762759323174886, 0.00418280011853364, 0.0006925062977488528, -0.003016419809918337, -0.0071049531892497, -0.011874829529214339, -0.017440963696104015, -0.023047513564523475, -0.02833190814159476, -0.03288842847284285, -0.03641158408109318, -0.037768893627003246, -0.03797559234861602, -0.03761714725398, -0.036702695029715245, -0.035227785335510846, -0.033104414148684154, -0.030095161935387575, -0.026321748778324498, -0.021683756060763528])


    construct_3dof(bd_yaml, node_interest, out_3dof, angle_attack,
                   load_grid, load_val, 
                   twist_grid, twist_val,
                   pitch_axis_grid, pitch_axis_val,
                   chord_grid, chord_val)


if __name__=="__main__":

    bd_yaml = os.path.join(my_dir, '../bd_driver.BD.sum.yaml')

    construct_IEA15MW_chord(bd_yaml, 'chord_3dof.yaml', 50, node_interest=7)


    # bd_rect = '../bd_driver_rect.BD.sum.yaml'
    # construct_rect(bd_rect, 'rect_3dof', 0.0, node_interest=7)

    construct_IEA15MW_chord(bd_yaml, 'IEA15_aoa5_3dof.yaml', 5.0, node_interest=7)

    construct_IEA15MW_chord(bd_yaml, 'IEA15_aoa0_3dof.yaml', 0.0, node_interest=7)
