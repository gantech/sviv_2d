import numpy as np


#####
# Define Parameter Ranges of Interest

modal_freq = np.array([0.5040, 0.6891, 4.088]) # Hz
modal_damp_fac = np.array([0.0048, 0.0048, 0.01]) # Fraction of Modal Critical Damping

strouhal_range = [0.13, 0.20] # Looks like exepected is 0.15-0.18
strouhal_plot = 0.165 #0.165 # Plot FRF assuming this is the true value
npoints_st = 30

cord = 1 # m

angle_of_attack = 60 # deg

##### 
# Create a matrix of reasonable options

strouhal_vec = np.linspace(strouhal_range[0], strouhal_range[1], npoints_st)

#####
# Calculate u_infity for the range of cases:
#
# pg23, eqn (10): of Bidadi et al 2023:
#     St = f * c * sin(alpha) / u_inf

# calculate u_inf for the bounds

u_inf_lower = modal_freq * cord * np.sin(angle_of_attack*np.pi/180) / strouhal_range[1]
u_inf_upper = modal_freq * cord * np.sin(angle_of_attack*np.pi/180) / strouhal_range[0]

if u_inf_upper[0] > u_inf_lower[1]:

    frac = (u_inf_upper[1] - u_inf_lower[0]) \
             / (u_inf_upper[0] - u_inf_lower[0] + u_inf_upper[1] - u_inf_lower[1])

    print('Fraction needed {}, estimated points: {}'.format(frac, frac*2*npoints_st))
    
    noverlap = 50
    print('Using {} points for overlap'.format(noverlap))

    u_inf = np.hstack((np.linspace(u_inf_lower[0], u_inf_upper[1], noverlap) ,
                       np.linspace(u_inf_lower[2], u_inf_upper[2], npoints_st)))

else:
    print('Case not implemented')


# # Generating u for each case independently
# u_inf = np.zeros((modal_freq.shape[0], strouhal_vec.shape[0]))
# 
# for freq_ind in range(modal_freq.shape[0]):
#     u_inf[freq_ind, :] = modal_freq[freq_ind] * cord * np.sin(angle_of_attack*np.pi/180) / strouhal_vec
# 
# print(u_inf)
# 
# 
# u_inf = u_inf.reshape(-1)

u_inf.sort()


print('u_inf [m/s]:')
print(u_inf)


#####
# Assuming some true Strouhal number, discretely
# plot the sample points of interest v. the 
# FRF of the structure - maybe just do it modally

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


# Reference points for the plot
u_inf_ref = np.linspace(u_inf.min(), u_inf.max(), 1000)

freq_forced_ref = strouhal_plot * u_inf_ref / cord / np.sin(angle_of_attack*np.pi/180)

# sampled forcing frequencies:
freq_forced = strouhal_plot * u_inf / cord / np.sin(angle_of_attack*np.pi/180)

# Loop and make plots
for freq_ind in range(modal_freq.shape[0]):

     # Sampled Points for u_inf
     r_force = freq_forced / modal_freq[freq_ind]
     
     amp = 1 / np.sqrt( (1 - r_force**2)**2 + (2*modal_damp_fac[freq_ind]*r_force)**2 )
     
     # Reference points for this plot
     r_force = freq_forced_ref / modal_freq[freq_ind]
     
     amp_ref = 1 / np.sqrt( (1 - r_force**2)**2 + (2*modal_damp_fac[freq_ind]*r_force)**2 )

     amp = amp / amp_ref.max()
     amp_ref = amp_ref / amp_ref.max()

     plt.plot(u_inf_ref, amp_ref)
     plt.plot(u_inf, amp, 'o')
     plt.xlabel('U infinity')
     plt.ylabel('Response Amplitude')
     plt.yscale('log')
     plt.title('Mode {}'.format(freq_ind+1))
     plt.grid('on')
     
     plt.savefig('mode_{}_frf.png'.format(freq_ind+1))


