import numpy as np


#####
# Define Parameter Ranges of Interest

modal_freq = np.array([0.5040, 0.6891, 4.088]) # Hz
modal_damp_frac = np.array([0.0048, 0.0048, 0.01]) # Fraction of Modal Critical Damping

strouhal_plot = 0.16 #0.165 # Plot FRF assuming this is the true value
npoints_st = 30

chord = 2.8480588747143942 # m

angle_of_attack = 50 # deg


##### 
# Choose some frequencies to run systematically
# These are in Hz

freq_test = np.array([0.5])

# First Test
# freq_test = np.copy(modal_freq)

# Second test, going for about half the displacement
freq_frac = 0.05
freq_test = np.hstack(( modal_freq * (1 - freq_frac), 
                        modal_freq * (1 + freq_frac),
                        modal_freq * (1 - 2*freq_frac), 
                        modal_freq * (1 + 2*freq_frac),
                        np.sum(modal_freq[0:2])/2 ))

freq_test.sort()

print('Test Frequencies [Hz]')
print(np.round(freq_test, 8).tolist(), sep=',')

##### 
# Calculate uinf based on frequencies choosen
# Then calculate Reynolds number
#
# pg23, eqn (10): of Bidadi et al 2023:
#     St = f * c * sin(alpha) / u_inf


uinf = freq_test * chord * np.sin(angle_of_attack*np.pi/180) / strouhal_plot 

print('u_inf [m/s]:')
print(uinf)


# Calculate Reynolds Number
rho = 1.225 # Density
viscosity = 1.0e-5

Re = rho * uinf * chord / viscosity

print('Re:')
print(Re)

############
# Misc old


#####
# Assuming some true Strouhal number, discretely
# plot the sample points of interest v. the 
# FRF of the structure - maybe just do it modally

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


# Reference points for the plot

freq_forced_ref = np.linspace(0.01, 1.5*modal_freq[-1], 1000)*np.pi*2.0

 
# sampled forcing frequencies:
freq_forced = freq_test*np.pi*2.0

# Loop and make plots
for freq_ind in range(modal_freq.shape[0]):

     # Sampled Points for u_inf
     r_force = freq_forced / (modal_freq[freq_ind] * np.pi * 2.0)
     
     amp = 1 / np.sqrt( (1 - r_force**2)**2 + (2*(2*np.pi*modal_damp_frac[freq_ind])*r_force)**2 )
     
     # Reference points for this plot
     r_force = freq_forced_ref / (modal_freq[freq_ind] * 2.0 * np.pi)
     
     amp_ref = 1 / np.sqrt( (1 - r_force**2)**2 + (2*(modal_damp_frac[freq_ind]*2*np.pi)*r_force)**2 )

     amp = amp / amp_ref.max()
     amp_ref = amp_ref / amp_ref.max()

     plt.plot(freq_forced_ref/2/np.pi, amp_ref)
     plt.plot(freq_forced/2/np.pi, amp, 'o')
     plt.xlabel('Frequency [Hz]')
     plt.ylabel('Response Amplitude')
     plt.yscale('log')
     plt.title('Mode {}'.format(freq_ind+1))
     plt.grid('on')
     
     plt.savefig('mode_{}_frf.png'.format(freq_ind+1))


