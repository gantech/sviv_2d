import netCDF4 as nc
import numpy as np

############

print('Run the nalu-wind input deck in this folder before running this python script.')

############
# File with the scaled loads (from SMD)
output_nc = 'af_smd_deflloads.nc'

# file with the unscaled loads from normal nalu-wind outputs
output_force = 'results/forces.dat'

##########
# Load the NC file

df = nc.Dataset(output_nc)

t_nc = np.array(df['time'][:])
f_nc = np.array(df['f'][:])

###########
# Load forces.dat

dat_out = np.genfromtxt(output_force, skip_header=1)

t_dat = dat_out[:, 0]
f_dat = np.hstack((dat_out[:, 1:3],  dat_out[:, 9:10]))

# print(f_dat)

###########
# Drop first row of the nc file since it starts at time 0 instead of the first step

t_nc = t_nc[1:]
f_nc = f_nc[1:, :]

############
# Calculate the load ratio

load_ratio = f_nc / f_dat

print('[Time, Load Ratio X, Y, MZ]')
print(np.hstack(( t_dat.reshape(-1,1), load_ratio)))

################
# Expected load fraction

exp_frac = (3 * ((t_nc-0.2) / 0.6)**2 - 2 * ( (t_nc - 0.2) / 0.6)**3 )
exp_frac[t_nc <= 0.2] = 0.0
exp_frac[t_nc >= 0.8] = 1.0

print('expected fraction:')
print(exp_frac)

################
# Error in fraction

ratio_error = np.abs(load_ratio - exp_frac.reshape(-1,1) ).max()

print('The maximum error in the ratio is: {}'.format(ratio_error))



