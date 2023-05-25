"""
Functions useful in constructing the 3 DOF model
Includes misc FEM type functions for different aspects
Also includes some modal analysis functions (eventually)
"""

import numpy as np
from numpy.polynomial import polynomial as p
from numpy.polynomial import Polynomial as Poly

import yaml
from yaml.loader import SafeLoader 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_M_K_nodes(yamlfile):
    """
    Load the Mass, Stiffness, and nodal coordinates

    Inputs:
      yamlfile - filename of .yaml file that contains the fields
    Outputs:
      Mmat - Mass Matrix, IEC coordinates
      Kmat - Stiffness Matrix, IEC coordinates
      node_coords - coordinates of the nodes (all 6 DOFs)
      quad_coords - coordinates of the quadrature points used
    """

    # Load YAML file
    with open(yamlfile) as f:
        data = list(yaml.load_all(f, Loader=SafeLoader))
    
    dict_data = data[-1]
    
    # print(dict_data.keys())
    
    Mmat = np.array(dict_data['M_IEC'])
    Kmat = np.array(dict_data['K_IEC'])
    node_coords = np.array(dict_data['Init_Nodes_E1'])
    quad_coords = np.array(dict_data['Init_QP_E1'])

    return Mmat, Kmat, node_coords, quad_coords



def construct_shape_funs(nodes):
    """
    Construct polynomials for the shape functions for the given nodes
    Shape functions are 1 at the node of the same index
    Shape functions are 0 at all other nodes

    nodes should be a 1D array with only the nodal coordinates 
    along the position  of interest
    """
    
    polys = nodes.shape[0] * [None]
    
    for i in range(nodes.shape[0]):

        # Set of nodes that are roots for the current shape function
        nodesTF = np.zeros(nodes.shape[0]) == 0
        nodesTF[i] = False

        # Polynomial shape function w/o normalization    
        polyi = Poly.fromroots(nodes[nodesTF])
        
        # Normalize to have nodal value of 1
        node_val = polyi(nodes[i]) 
        polyi = polyi / node_val
    
        polys[i] = polyi
    
    return polys

def calc_trap_weights(quad_coords, refine=False):
    """
    Calculate quadrature weights associated with trapezoid rule and
    the quadrature coordinates. Assume no refinement between coordinates. 

    Inputs:
      quad_coords - coordinates for the quadrature points (6 DOF)
      refine - if true, splits trapezoids into 2 sections.

    Outputs:
      weights - trapezoid rule weights for points
    """

    span_coords = quad_coords[:, 2]

    if refine:
        span_coords = np.hstack((span_coords, \
                            (span_coords[1:] + span_coords[:-1])/2.0))
        
        span_coords.sort()
    

    weights = np.zeros_like(span_coords)

    weights[0] = (span_coords[1] - span_coords[0])/2.0
    weights[-1] = (span_coords[-1] - span_coords[-2])/2.0

    weights[1:-1] = (span_coords[2:] - span_coords[0:-2]) / 2.0

    trap_coords = np.copy(span_coords)
    return trap_coords,weights

def construct_trap_int_mat(node_coords, quad_coords, refine=False):
    """
    Construct an integration matrix from points along the blade 
    to nodal forces. 

    Inputs:
      node_coords - coordinates of beam element nodes
      quad_coords - coordinates of the key points on the blade
      refine - if coordinates should be added between each quad_coord

    Outputs:
      x_traps - locations of quadrature points to evaluate (span location)
      int_mat_trap - integration matrix that transforms from x_traps to nodes
    """

    # Make shape functions for elements
    nodes = node_coords[:, 2]
    polys = construct_shape_funs(nodes)

    # Construct quadrature rule
    x_traps, w_traps = calc_trap_weights(quad_coords, refine=refine)

    # Initialize matrix
    int_mat_trap = np.zeros((nodes.shape[0], w_traps.shape[0]))

    for i in range(nodes.shape[0]):
        
        shape_fun = polys[i](x_traps)
    
        int_mat_trap[i, :] = shape_fun * w_traps

    return x_traps, int_mat_trap


def gram_schmidt(Klocal, local_phi, local_eigs, ordering=[0,1,2]):
    """
    Update the local eigenvectors to be orthogonal w.r.t. the local stiffness matrix
    In addition, phi^T K phi = diag(local_eigs) for normalizing new vectors
    Doing this step guarantees that when the mass matrix is constructed, 
    the eigenvectors and eigenvalues will exactly match those used in construction

    Inputs:
      Klocal - 3 x 3 stiffness matrix for the 3 DOF model
      local_phi - desired mode shapes (not orthogonal)
      local_eigs - desired eigenvalues (frequency in rad/s squared)
      ordering - order that eigenvectors should be made orthogonal
                  the first eigenvector in the order is exactly preserved
                  the third eigenvector will likely have the most change

    Outputs:
      ortho_phi - eigenvectors that are based on local_phi, but are 
                  orthogonal w.r.t. Klocal
    """
    
    ortho_phi = np.copy(local_phi)
    local_eigs = np.copy(local_eigs)

    # reorder columns and eigenvalues
    ortho_phi = ortho_phi[:, ordering]
    local_eigs = local_eigs[ordering]
    ordering_reset = np.array(range(3))[ordering]


    for i in range(1,3):
        for j in range(i):

            proj = (ortho_phi[:, i:i+1].T @ Klocal @ ortho_phi[:, j:j+1])\
                    / (ortho_phi[:, j:j+1].T @ Klocal @ ortho_phi[:, j:j+1])\
                    * ortho_phi[:, j:j+1]

            ortho_phi[:, i:i+1] -= proj

    # Rescale modes to have correct normalization
    local_scale = np.diag(ortho_phi.T @ Klocal @ ortho_phi)
    ortho_phi = ortho_phi * np.sqrt(local_eigs/local_scale)

    # Reset ordering
    ortho_phi[:, ordering] = ortho_phi # [:, ordering_reset]

    return ortho_phi


def plot_nodal_field(nodes, vals, filename, title='', ylabel='', xlabel='Span Position'):
    """
    Construct a plot of some nodal value over
    the length of the blade

    inputs:
      nodes - nodal coordinates
      vals - values of the field of interest at nodes
      filename - filename to save output as
      title - title to put on plot
    """
    
    polys = construct_shape_funs(nodes)

    x_vals = np.linspace(nodes[0], nodes[-1], 1000)

    y_vals = np.zeros_like(x_vals)

    # Evaluate interpolating polynomials
    for i in range(nodes.shape[0]):
        y_vals += polys[i](x_vals) * vals[i]

    # Plot the results
    plt.plot(x_vals, y_vals)
    plt.xlim((nodes[0], nodes[-1]))
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.savefig(filename)
    plt.close()

    return

