"""
Plot the Mode Shapes of the Structure for inclusion in presentation / paper

Also plot the chord distribution / load distribution as well
"""

import numpy as np
import matplotlib.pyplot as plt

import yaml

import sys
sys.path.append('..')
import construct_utils as cutils


#########################################
# Inputs

bd_yaml = '../bd_driver.BD.sum.yaml'


node_interest = 7 # node of interest in python index.


mode_indices = [0, 1, 5] # Mode indices for first flap, edge, twist modes.
disp_dirs = [0, 1, 5] # Directions of displacements to plot

# Chord Loading Parameters
# Consider loads proportional to the chord length
# chord_grid - points where chord is defined (fraction of span length)
# chord_val - values of chord length

chord_grid = [0.0, 0.02040816326530612, 0.04081632653061224, 0.061224489795918366, 0.08163265306122448, 0.1020408163265306, 0.12244897959183673, 0.14285714285714285, 0.16326530612244897, 0.18367346938775508, 0.2040816326530612, 0.22448979591836732, 0.24489795918367346, 0.26530612244897955, 0.2857142857142857, 0.3061224489795918, 0.32653061224489793, 0.3469387755102041, 0.36734693877551017, 0.3877551020408163, 0.4081632653061224, 0.42857142857142855, 0.44897959183673464, 0.4693877551020408, 0.4897959183673469, 0.5102040816326531, 0.5306122448979591, 0.5510204081632653, 0.5714285714285714, 0.5918367346938775, 0.6122448979591836, 0.6326530612244897, 0.6530612244897959, 0.673469387755102, 0.6938775510204082, 0.7142857142857142, 0.7346938775510203, 0.7551020408163265, 0.7755102040816326, 0.7959183673469387, 0.8163265306122448, 0.836734693877551, 0.8571428571428571, 0.8775510204081632, 0.8979591836734693, 0.9183673469387754, 0.9387755102040816, 0.9591836734693877, 0.9795918367346939, 0.985, 0.99, 0.995, 1.0]    

chord_val  = [5.2, 5.208839941579524, 5.237887092263203, 5.293325313383697, 5.3673398548149205, 5.452092684226667, 5.5400317285038465, 5.621824261194381, 5.692531175149338, 5.74261089072697, 5.764836827022541, 5.756119529852528, 5.70309851275065, 5.604676021602162, 5.471559126660524, 5.322778014171772, 5.16648228816705, 5.019421327310202, 4.885807888739599, 4.767959675121795, 4.654566079625438, 4.54103105171191, 4.42817557762473, 4.316958876583997, 4.207880735790049, 4.101646187027423, 3.9987123353123564, 3.8994086760515647, 3.803172543681295, 3.7093894536544263, 3.6171117415725322, 3.525634918177657, 3.434082670567315, 3.341933111457596, 3.2486784477614132, 3.156109679927359, 3.0645800048338336, 2.9729926470824872, 2.8807051066906166, 2.786969376686517, 2.6910309386270574, 2.591965555977676, 2.4893236475052167, 2.383917231097341, 2.2759238162069977, 2.165466732053696, 2.0526250825584818, 1.9377533268191636, 1.819662967336163, 1.7799216728668075, 1.7077871948468315, 1.472482673397968, 0.5000000000000001]


load_span = np.array(chord_grid)
load_val = np.array(chord_val)


#########################################
# Modal Analysis

Mmat, Kmat, node_coords, quad_coords = cutils.load_M_K_nodes(bd_yaml)


# Apply boundary conditions
Mbc = Mmat[6:, 6:]
Kbc = Kmat[6:, 6:]

eigvals,eigvecs = cutils.modal_analysis(Kbc, Mbc)


#########################################
# Interpolate with FEM shape functions

nodes = node_coords[:, 2]


modal_shapes = len(mode_indices) * [None]



# mode_indices = [0, 1, 5] # Mode indices for first flap, edge, twist modes.
# disp_dirs = [0, 1, 5] # Directions of displacements to plot

for mode_i in range(len(mode_indices)):
    
    mode_dir_disp = len(disp_dirs) * [None]

    mode_shape = eigvecs[:, mode_indices[mode_i]]

    for dir_i in range(len(disp_dirs)):

        nodal_phys = np.hstack(([0], mode_shape[disp_dirs[dir_i]::6]))

        span_pos,disp_mode = cutils.interpolate_nodal_field(nodes, nodal_phys)

        mode_dir_disp[dir_i] = disp_mode

    modal_shapes[mode_i] = mode_dir_disp



#########################################
# Plots of mode shapes

plt.style.use('seaborn-v0_8-colorblind') 


labels = ['Flap Mode', 'Edge Mode', 'Twist Mode']
ylabels = ['Flap [m]', 'Edge [m]', 'Twist [deg]']
disp_convert = [1.0, 1.0, 180/np.pi]

symbol = ['-', '--', '-.']

fig,axs = plt.subplots(3)

lw=2

for dir in range(3):

    for mode in range(3):

        axs[dir].plot(span_pos, modal_shapes[mode][dir]*disp_convert[dir], symbol[mode], label=labels[mode], linewidth=lw)

    axs[dir].set_ylabel(ylabels[dir])

    axs[dir].set_xlim((0, span_pos[-1]))

    axs[dir].tick_params(bottom=True, top=True, left=True, right=True,direction="in")

# Only label the bottom x axis
axs[0].xaxis.set_tick_params(labelbottom=False)
axs[1].xaxis.set_tick_params(labelbottom=False)

axs[2].set_xlabel('Spanwise Position [m]')

# Only legend on top
axs[0].legend()

fig.tight_layout()
fig.subplots_adjust(hspace=0.04)


plt.savefig('ModeShapes.png', dpi=300)
plt.close()

#########################################
# Plot of load distribution

fig = plt.figure(figsize = [6,2.5])

load_scale = np.interp(nodes[node_interest] / nodes[-1], load_span, load_val, left=np.nan, right=np.nan)


plt.plot(load_span*nodes[-1], load_val/load_scale)

plt.xlim((0, span_pos[-1]))

plt.ylabel('Load Distribution [N/m/N]')
plt.xlabel('Spanwise Position [m]')

fig.tight_layout()
plt.savefig('LoadShape.png', dpi=300)
plt.close()

