"""
Functions for Peak Finding and Fitting Algorithm 
This algorithm achieves similar results to a Hilbert Transform, but has less end effects


The paper for this algorithm is:

M. Jin, W. Chen, M.R.W. Brake, H. Song,
Identification of Instantaneous Frequency and Damping From Transient Decay Data, 
Journal of Vibration and Acoustics, 2020.


Algorithm is slightly modified since the behavior of interest is at the end of the simulation 
rather than the beginning. 

The current implementation may not include all parts of the algorithm. 
"""

import numpy as np
import netCDF4 as nc

from scipy import signal

import matplotlib.pyplot as plt

def identify_peak_freq(t, x, nom_freq, tstart, search_bandwidth=0.2):
    """
    Identify the peak frequency near the nominal frequency to use as the center for the filter
    """

    dt = t[1] - t[0]
    fs = 1 / dt

    x = np.copy(x)
    x[t < tstart, :] = 0.0

    Xfft = np.abs(np.fft.fft(x, axis=0))
    freq = fs*np.fft.fftfreq(x.shape[0])

    freq_mask = np.logical_and( freq > (1 - search_bandwidth)*nom_freq, \
                                freq < (1 + search_bandwidth)*nom_freq )
   
    max_ind = np.argmax(Xfft[freq_mask, :], 0)

    freq_peak = freq[freq_mask][max_ind]

    # Just return the median for now to filter all motions to the same freq. 
    # Prevents risk of outliers causing problems
    freq_peak = np.median(freq_peak)

    return freq_peak



def double_filter(t, x, center_freq, half_bandwidth_frac, tstart, npad=500, filter_order=3):
    """
    Apply the double reverse filtering algorithm 
    Since the signal of interest is at the end of the time series, 
    filtering is first done forwards then reverse. 
    """

    dt = t[1] - t[0]
    fs = 1 / dt

    x = np.copy(x)
    x = x[t >= tstart, :]

    x = np.vstack((np.zeros((npad, x.shape[1])), x, np.zeros((npad, x.shape[1]))))

    filter_points = [(1 - half_bandwidth_frac)*center_freq, \
                     (1 + half_bandwidth_frac)*center_freq ]

    sos = signal.butter(filter_order, filter_points, btype='bandpass', fs=fs, output='sos')

    x_filt = signal.sosfiltfilt(sos, x, axis=0)

    # remove the zero padding
    if npad > 0:
        x_filt = x_filt[npad:-npad, :]

    return x_filt


def fit_peaks(t, xcol):
    """
    Fit peaks to sets of 3 points for maxima or minimu

    Inputs:
      t - time series
      xcol - single 1D vector of data to fit
    """

    peaks = np.logical_and( xcol > np.roll(xcol, 1), xcol > np.roll(xcol, -1) )

    peaks[0] = False
    peaks[-1] = False


    # Fit quadratic to three points here
    t0 = t[np.roll(peaks, -1)].reshape((1,1,-1))
    t1 = t[peaks].reshape((1,1,-1))
    t2 = t[np.roll(peaks, 1)].reshape((1,1,-1))


    x0 = xcol[np.roll(peaks, -1)].reshape((1,1,-1))
    x1 = xcol[peaks].reshape((1,1,-1))
    x2 = xcol[np.roll(peaks, 1)].reshape((1,1,-1))

    # Construct matrices to solve
    rhs = np.vstack((x0, x1, x2))

    one_vec = np.ones(t0.shape)

    lhs = np.vstack(( np.hstack((t0**2, t0, one_vec )), \
                      np.hstack((t1**2, t1, one_vec )), \
                      np.hstack((t2**2, t2, one_vec )) ))

    # identify peaks times and amplitudes
    amp_val   = np.zeros(peaks.sum()) 
    peak_time = np.zeros(peaks.sum()) 

    for i in range(amp_val.shape[0]):

        coefs = np.linalg.solve(lhs[:, :, i], rhs[:, 0, i])

        peak_time[i] = -1*coefs[1] / 2.0 / coefs[0]

        amp_val[i] = coefs[0]*peak_time[i]**2 + coefs[1]*peak_time[i] + coefs[2]

    
    return peak_time, amp_val

def peak_max_min_id(t, x):
    """
    Identify both the maxima and minima of the time series
    """


    # create lists for stored peak values and mimima
    peak_times  = x.shape[1] * [None]
    peak_values = x.shape[1] * [None]

    # Loop over columns
    for col_ind in range(x.shape[1]):

        xcol = x[:, col_ind]

        peak_time_max, peak_val_max = fit_peaks(t, xcol)
        peak_time_min, peak_val_min = fit_peaks(t, -xcol)

        peak_val_min = -1*peak_val_min

        peak_time_col = np.hstack((peak_time_max, peak_time_min))
        inds = np.argsort(peak_time_col)

        peak_times[col_ind] = peak_time_col[inds]
        peak_values[col_ind] = np.hstack((peak_val_max, peak_val_min))[inds]

    return peak_times, peak_values


def calc_freq_damp(peak_times, peak_values):
    """
    Calculate the frequency and damping values for the instants in time
    """

    freq_rad_s = len(peak_times)*[None]
    damp_frac_crit = len(peak_times)*[None]

    report_amp = len(peak_times)*[None]
    report_t = len(peak_times)*[None]

    for col_ind in range(len(peak_times)):

        freq_rad_s[col_ind] = np.pi / np.diff(peak_times[col_ind])[1:]
        
        report_t[col_ind] = peak_times[col_ind][1:-1]
        report_amp[col_ind] = peak_values[col_ind][1:-1]

        damp_frac_crit[col_ind] = -1.0/freq_rad_s[col_ind] \
                            *( np.log(np.abs(np.roll(peak_values[col_ind],-1))) - np.log(np.abs(np.roll(peak_values[col_ind],1))) )[1:-1] \
                            / ( np.roll(peak_times[col_ind],-1) - np.roll(peak_times[col_ind],1) )[1:-1]


    return freq_rad_s, damp_frac_crit, report_t, report_amp


def eigen_realization_alg(xcol, ncreate, nSRM=1, preserveVals=2):
    """
    Extrapolate the signal xcol for ncreate more points

    Augments the signal towards earlier times because that is the algorithm
    that is written out in the PFF paper. 

    Inputs:
      nSRM - relates to the order of the extrapolation, 1 is probably fine.
      preserveVals - number of singular values to use for extrapolation. PFF paper uses 2*nSRM
    """

    # Zero mean the data
    meanx = np.mean(xcol)

    xcol = xcol - meanx

    Hrows = 2*nSRM + 2
    Hcols = xcol.shape[0] - Hrows

    # Construt Hankel Matrices (0 and 1)
    H0 = np.zeros((Hrows, Hcols))
    H1 = np.zeros((Hrows, Hcols))

    for row in range(Hrows):
        for col in range(Hcols):
            H0[row, col] = xcol[row+col]
            H1[row, col] = xcol[row+col+1]

    # Apply extrapolation algorithm
    U,s,Vh = np.linalg.svd(H0)

    Ured = U[:, :preserveVals]
    Sred = np.diag(s[:preserveVals])
    Sred_halfinv = np.diag(s[:preserveVals]**(-0.5))
    Vhred = Vh[:preserveVals, :]

    E1 = np.zeros(Hcols)
    E1[0] = 1

    Em = np.zeros(Hrows)
    Em[0] = 1

    # Transpose on Vhred appears to be wrong in paper, and corrected here after checking their code.
    A  = Sred_halfinv @ Ured.T @ H1 @ Vhred.T @ Sred_halfinv
    C  = Em.T @ Ured @ (Sred**0.5)
    x0 = (Sred**0.5) @ Vhred @ E1

    xnew = np.zeros(ncreate)
    xi = x0

    Ainv = np.linalg.inv(A)

    for i in range(ncreate):
        xi = Ainv @ xi

        xnew[-i] = C @ xi
    
    # undo zero mean
    xaug = np.hstack((xnew, xcol)) + meanx
    return xaug

    


def expand_signal(t, x, nbasis, nforward, nbackward):
    """
    Augment signals on both sides with extrapolation algorithm before filtering
    """

    augmented_x = np.zeros((x.shape[0]+nforward+nbackward, x.shape[1]))    

    for col_ind in range(x.shape[1]):
        
        xbasis = x[:nbasis, col_ind]
        xaug = eigen_realization_alg(xbasis, nbackward)
        xstart = xaug[:nbackward]
        
        xbasis = x[-nbasis:, col_ind]
        xbasis = xbasis[::-1]
        xaug = eigen_realization_alg(xbasis, nforward)
        
        xend = xaug[:nforward]
        xend = xend[::-1]

        augmented_x[:, col_ind] = np.hstack((xstart, x[:, col_ind], xend))

    tstart = t[0]
    tend = t[-1]


    dt = t[1] - t[0]
    
    augmented_t = np.hstack((-dt*np.array(range(nbackward, 0, -1))+t[0], \
                             t, \
                             dt*np.array(range(1, nforward+1, 1))+t[-1]))

    return augmented_x, augmented_t, tstart, tend


def pff_analysis(t, x, nom_freq, ttrim, half_bandwidth_frac, remove_end=0):
    """
    Conduct a PFF analysis with filtering

    Inputs:
      t - time series points
      x - displacements with rows corresponding to times t and columns for different DOFs
      nom_freq - the nominal frequency to conduct the calculation around
      ttrim - start time to cut the signal to before any processing is conducted
      half_bandwidth_frac - fraction of the peak frequency to use as half bandwidth in filtering
      remove_end - number of points to eliminate from the end of the signal. 
         Usually due to end effects that are more prominent for multiharmonic signals after filtering. 
         This number of points is removed from both the start and the end of the signal.

    Outputs:
      freq_rad_s - frequency in radians per second
      damp_frac_crit - fraction of critical damping 
      report_t - time instants for frequency and damping values
      report_amp - amplitudes at the reported time instants
      intermediate_data - tuple of various intermediate states for plotting and visual checks
    """

    # Cut off times before tstart
    x = x[t >= ttrim, :]
    t = t[t >= ttrim]

    # Help the algorithm by moving x to have zero mean from the start
    x = x - np.mean(x, axis=0)

    # Identify the main frequency in the data
    peak_freq = identify_peak_freq(t, x, nom_freq, 0)

    # Extend both ends of the data by approximately 5 cycles using 5 cycles as a basis
    dt = t[1] - t[0]
    nbasis = int(5.0/peak_freq / dt) + 1

    augmented_x, augmented_t, tstart, tend = expand_signal(t, x, nbasis, nbasis, nbasis)

    # Filter the data
    x_filt = double_filter(augmented_t, augmented_x, peak_freq, half_bandwidth_frac, -np.inf, npad=0)

    # Identify peaks
    peak_times, peak_val = peak_max_min_id(augmented_t, x_filt)

    # Identify nonlinear freq and damping
    freq_rad_s, damp_frac_crit, report_t, report_amp = calc_freq_damp(peak_times, peak_val)

    # Remove any spurious points from freq and damping data from extrapolated times
    for col_ind in range(len(freq_rad_s)):
        mask = np.logical_and(report_t[col_ind] > tstart, report_t[col_ind] < tend)

        if mask.sum() > (2 + 2*remove_end):
            # one data point must be removed from each end because the finite difference
            # uses extrapolated data for the first and last point. 

            freq_rad_s[col_ind]         = freq_rad_s[col_ind][mask][1+remove_end:-1-remove_end]
            damp_frac_crit[col_ind] = damp_frac_crit[col_ind][mask][1+remove_end:-1-remove_end]
            report_t[col_ind] =             report_t[col_ind][mask][1+remove_end:-1-remove_end]
            report_amp[col_ind] =         report_amp[col_ind][mask][1+remove_end:-1-remove_end]

        else:  
            freq_rad_s[col_ind] = np.zeros(0) 
            damp_frac_crit[col_ind] = np.zeros(0)
            report_t[col_ind] = np.zeros(0)
            report_amp[col_ind] = np.zeros(0)

    # Return Several extra things for debugging / verification
    intermediate_data = (t, x, augmented_t, augmented_x, x_filt, peak_times, peak_val)

    return freq_rad_s, damp_frac_crit, report_t, report_amp, intermediate_data

def plot_pff_results(freq_rad_s, damp_frac_crit, report_t, report_amp, intermediate_data, base=''):


    t, x, augmented_t, augmented_x, x_filt, peak_times, peak_val = intermediate_data

    #########
    # Plot Settings
    plt.style.use('seaborn-v0_8-colorblind') 



    ####################
    # Plots for verification and such of results
    # Stability application is for at end of time series, so look at those

    label = ['x [m]', 'y [m]', 'theta [deg]']
    convert_unit = [1.0, 1.0, 180.0/np.pi]

    for dir in range(len(freq_rad_s)):

        plt.close()

        ###
        # Original Signal (trimmed)
        plt.plot(t, x[:, dir]*convert_unit[dir], label='Original')

        # Extend Signal Algorithm
        plt.plot(augmented_t, augmented_x[:, dir]*convert_unit[dir], '--', label='Augmented')

        # Filtering Portion
        plt.plot(augmented_t, x_filt[:, dir]*convert_unit[dir], ':', label='Filtered')

        # Peaks Identification
        plt.plot(report_t[dir], report_amp[dir]*convert_unit[dir], 'o', markerfacecolor='none', label='Peaks')

        plt.legend(loc='center left', framealpha=1.0)
        plt.xlim((augmented_t[0], augmented_t[-1]))

        plt.xlabel('Time [s]')
        plt.ylabel(label[dir])

        plt.savefig('overview_pff{}_dir{}.png'.format(base,dir), dpi=300)

        plt.close()

    ##############
    # Plot Damping and Frequency

    freq_damp_labels = ['x', 'y', 'theta']

    fig, axs = plt.subplots(2)

    # print(report_t)

    min_t = report_t[0][0]
    max_t = report_t[0][-1]

    for dir in range(len(freq_rad_s)):

        axs[0].plot(report_t[dir], freq_rad_s[dir]/2/np.pi, 'o', markerfacecolor='none', label=freq_damp_labels[dir])


        axs[1].plot(report_t[dir], damp_frac_crit[dir], 'o', markerfacecolor='none', label=freq_damp_labels[dir])

        min_t = np.minimum(min_t, report_t[dir].min())
        max_t = np.maximum(max_t, report_t[dir].max())


    axs[0].legend(loc='best')

    extra_t = np.pi/freq_rad_s[dir].mean()

    for ax in axs:
        ax.tick_params(bottom=True, top=True, left=True, right=True,direction="in")
        ax.set_xlim((min_t-extra_t, max_t+extra_t))
        
    # only label bottom time axis
    axs[0].xaxis.set_tick_params(labelbottom=False)
    axs[1].set_xlabel('Time [s]')

    axs[0].set_ylabel('Frequency [Hz]')
    axs[1].set_ylabel('Fraction Critical Damping')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.04)

    plt.savefig('freq_damp_time{}.png'.format(base), dpi=300)
    plt.close()



if __name__=="__main__":

    # Inputs

    import_flag = 1

    if import_flag == 0:

        filename = 'af_smd_deflloads.nc'

        data = nc.Dataset(filename)
        
        t = np.array(data['time'][:])
        x = np.array(data['x'][:])
        f = np.array(data['f'][:])
    elif import_flag == 1:

        filename = 'ua_test.npz'

        npzfile = np.load(filename)

        x = npzfile['x']
        f = np.copy(x)
        t = npzfile['t']

        print(x.shape)
        print(t.shape)



    tstart = 20
    nom_freq = 0.7
    half_bandwidth_frac = 0.2
    remove_end = 0

    ###############
    # Example of full process on real data

    freq_rad_s, damp_frac_crit, report_t, report_amp, intermediate_data = \
         pff_analysis(t, x, nom_freq, tstart, half_bandwidth_frac, remove_end=remove_end)

    plot_pff_results(freq_rad_s, damp_frac_crit, report_t, report_amp, intermediate_data)

    ###############
    # run PFF on the force data to analyze how forces are changing 

    freq_rad_s, damp_frac_crit, report_t, report_amp, intermediate_data = \
         pff_analysis(t, f, nom_freq, tstart, half_bandwidth_frac, remove_end=remove_end)

    plot_pff_results(freq_rad_s, damp_frac_crit, report_t, report_amp, intermediate_data, base='_force')


    ###############
    # Example on artificial data for verification

    dt = 0.01
    tmax = 50
    t_verify = np.array(range(0, int(tmax / dt)))*dt
    freq = 0.5 # Hz
    amp = 2
    damp_crit = 0.01

    omega = freq * np.pi * 2
    omega_damped = omega * np.sqrt(1 - damp_crit**2)

    x_verify = np.ones((3,1)) * (amp*np.exp(-damp_crit*omega*t_verify)*np.cos(omega_damped*t_verify) )

    x_verify = x_verify.T

    # Add some noise to a few columns of x
    x_verify[:, 1] = x_verify[:, 0] + np.cos(omega_damped*1.75*t_verify)*amp/6
    x_verify[:, 2] = x_verify[:, 0] + np.cos(omega_damped*0.2*t_verify)*amp/6


    # Removing 0 demonstrates clear end effects
    # Removing 7 points from the ends eliminates these and is much better quality. 
    freq_rad_s, damp_frac_crit, report_t, report_amp, intermediate_data = \
         pff_analysis(t_verify, x_verify, nom_freq, tstart, half_bandwidth_frac, remove_end=7)

    plot_pff_results(freq_rad_s, damp_frac_crit, report_t, report_amp, intermediate_data, base='_verify')
