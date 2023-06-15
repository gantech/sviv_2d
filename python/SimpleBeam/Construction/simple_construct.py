import numpy as np

import sys
sys.path.append('../../ConstructModel')
sys.path.append('../..')

import create_model_funs as create_funs
import construct_utils as cutils

from scipy.linalg import eigh

def construct_rect(bd_yaml, out_3dof, angle_attack, node_interest=7):
    """
    Construct a 3 DOF model of a beam with a constant rectangular cross section.
    
    This function defines inputs to construct_3dof for the specific turbine. 
    See that function for input documentation. 
    """

    # chord calculation grid.
    chord_grid = [0.0, 1.0]    
    chord_val  = [4.0, 4.0]

    # Load distribution grid    
    load_grid = np.array(chord_grid)
    load_val  = np.array(chord_val)

    # Pitch axis offset grid
    pitch_axis_grid = [0.0, 1.0]     
    pitch_axis_val  = [0.0, 0.0]

    twist_grid = np.array([0.0, 1.0])
    twist_val = np.array([0.0, 0.0])

    create_funs.construct_3dof(bd_yaml, node_interest, out_3dof, angle_attack,
                   load_grid, load_val, 
                   twist_grid, twist_val,
                   pitch_axis_grid, pitch_axis_val,
                   chord_grid, chord_val,
                   mode_indices=[0,1,5])



if __name__=="__main__":

    bd_yaml = '../BeamDyn/bd_simple_driver.BD.sum.yaml'

    construct_rect(bd_yaml, 'SimpleBeam_3DOF.yaml', 0.0, node_interest=7)


    Mmat, Kmat, node_coords, quad_coords = cutils.load_M_K_nodes(bd_yaml)
    Mbc = Mmat[6:, 6:]
    Kbc = Kmat[6:, 6:]

    eigvals,eigvecs = cutils.modal_analysis(Kbc, Mbc)

    # print('Mode shapes - at tip - verifies the mode indices for flap,edge,twist')
    # print(eigvecs[-6:, :6])

    print('\nFrequencies [Hz]')
    print(np.sqrt(eigvals) / 2/ np.pi)

    print('Eigenvalues')
    print(eigvals)

    Fload = np.zeros_like(Kbc[:, 0])
    Fload[-1] = 10000

    static_disp = np.linalg.solve(Kbc, Fload)
    
    print('Static displacements for tip')
    print(static_disp[-6:])
