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

    # print(freq_peak)

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

    print(x.shape)

    x = np.vstack((np.zeros((npad, x.shape[1])), x, np.zeros((npad, x.shape[1]))))

    print(x.shape)


    filter_points = [(1 - half_bandwidth_frac)*center_freq, \
                     (1 + half_bandwidth_frac)*center_freq ]

    sos = signal.butter(filter_order, filter_points, btype='bandpass', fs=fs, output='sos')

    x_filt = signal.sosfiltfilt(sos, x, axis=0)

    print(x_filt.shape)

    # remove the zero padding
    if npad > 0:
        x_filt = x_filt[npad:-npad, :]

    print(x_filt.shape)

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


    print(peaks.sum())

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

    # print(rhs.shape)
    # print(lhs.shape)

    # identify peaks times and amplitudes
    amp_val   = np.zeros(peaks.sum()) 
    peak_time = np.zeros(peaks.sum()) 

    for i in range(amp_val.shape[0]):

        coefs = np.linalg.solve(lhs[:, :, i], rhs[:, 0, i])

        peak_time[i] = -1*coefs[1] / 2.0 / coefs[0]

        amp_val[i] = coefs[0]*peak_time[i]**2 + coefs[1]*peak_time[i] + coefs[2]

        # print('\ni = {}'.format(i))
        # print('Times = ({}, {}, {})'.format(t0[0,0,i], t1[0,0,i], t2[0,0,i]))
        # print('Xvals = ({}, {}, {})'.format(x0[0,0,i], x1[0,0,i], x2[0,0,i]))
        # print('Peak (t, x) = ({}, {})'.format(peak_time[i], amp_val[i]))


    
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

        print('peak_max_min_id')
        print(xcol.shape)

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

        freq_rad_s[col_ind] = np.pi / np.diff(peak_times[col_ind])
        
        report_t[col_ind] = peak_times[col_ind][:-1]
        report_amp[col_ind] = peak_values[col_ind][:-1]

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

    print('Aug time')
    print(augmented_t[:5])
    print(augmented_t[nbackward-3:nbackward+3])
    print(augmented_t[-nforward-3:-nforward+3])
    print(augmented_t[-5:])

    return augmented_x, augmented_t, tstart, tend


def pff_analysis(t, x, nom_freq, ttrim, half_bandwidth_frac):
    """
    Conduct a PFF analysis with filtering

    Inputs:
      t - time series points
      x - displacements with rows corresponding to times t and columns for different DOFs
      nom_freq - the nominal frequency to conduct the calculation around
      ttrim - start time to cut the signal to before any processing is conducted
      half_bandwidth_frac - fraction of the peak frequency to use as half bandwidth in filtering
    """

    # Cut off times before tstart
    x = x[t >= ttrim, :]
    t = t[t >= ttrim]

    # Identify the main frequency in the data
    peak_freq = identify_peak_freq(t, x, nom_freq, 0)

    # Extend both ends of the data by approximately 5 cycles using 5 cycles as a basis
    dt = t[1] - t[0]
    nbasis = int(1/peak_freq / dt) + 1
    print('nbasis = {}'.format(nbasis))

    augmented_x, augmented_t, tstart, tend = expand_signal(t, x, nbasis, nbasis, nbasis)

    # Filter the data
    x_filt = double_filter(augmented_t, augmented_x, peak_freq, half_bandwidth_frac, 0, npad=0)

    # Identify peaks
    print(x_filt.shape)
    peak_times, peak_val = peak_max_min_id(augmented_t, x_filt)

    # Identify nonlinear freq and damping
    freq_rad_s, damp_frac_crit, report_t, report_amp = calc_freq_damp(peak_times, peak_val)

    # Remove any spurious points from freq and damping data from extrapolated times
    for col_ind in range(len(freq_rad_s)):
        mask = np.logical_and(report_t[col_ind] < tstart, report_t[col_ind] > tend)

        freq_rad_s[col_ind] = freq_rad_s[col_ind][mask]
        damp_frac_crit[col_ind] = None # damp_frac_crit[col_ind][mask]
        report_t[col_ind] = report_t[col_ind][mask]
        report_amp[col_ind] = report_amp[col_ind][mask]

    # Return Several extra things for debugging / verification

    return freq_rad_s, damp_frac_crit, report_t, report_amp, augmented_x, augmented_t, x_filt

if __name__=="__main__":

    # Inputs

    filename = 'af_smd_deflloads.nc'

    data = nc.Dataset(filename)
    
    t = np.array(data['time'][:])
    x = np.array(data['x'][:])

    nom_freq = 0.7
    tstart = 10


    # Augment signal
    aug_dir = 0

    xend = x[t > 40.0, aug_dir]
    ncreate = xend.shape[0]

    xaug = eigen_realization_alg(xend[::-1], ncreate)
    xaug = xaug[::-1]

    # Run PFF and plot some comparisons for a specific case

    print(x.shape)

    peak_freq = identify_peak_freq(t, x, nom_freq, tstart)

    x_filt = double_filter(t, x, peak_freq, 0.2, tstart)

    # col = 0
    # peak_time, peak_val = fit_peaks(t[t >= tstart], x_filt[:, col])
    
    peak_times, peak_val = peak_max_min_id(t[t >= tstart], x_filt)

    freq_rad_s, damp_frac_crit, report_t, report_amp = calc_freq_damp(peak_times, peak_val)

    print(freq_rad_s[0]/2/np.pi)
    print(freq_rad_s[1]/2/np.pi)
    print(freq_rad_s[2]/2/np.pi)

    ###### Plot some comparisons to the original signal with the new signal


    import matplotlib.pyplot as plt

    for dir in range(3):

        plt.plot(t, x[:, dir], label='Original')

        mean_offset = np.mean(x[t >= tstart, dir])
        plt.plot(t[t >= tstart], x_filt[:, dir]+mean_offset, '--', label='Filtered + Mean')

        plt.plot(peak_times[dir], peak_val[dir]+mean_offset, 'o', label='ID Peak', markerfacecolor='none')

        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Amplitude in Direction {}'.format(dir))

        plt.savefig('FilteredCompare_dir{}.png'.format(dir))
        plt.close()


    plt.plot(xend, label='Original X')
    plt.plot(xaug, '--', label='Extrapolated X')

    plt.legend()
    plt.savefig('Extrapolated.png')
    plt.close()

    print(xaug[-10:])
    print(xaug.shape)
    print(xend.shape)


    half_bandwidth_frac = 0.2

    freq_rad_s, damp_frac_crit, report_t, report_amp, augmented_x, augmented_t, x_filt = \
         pff_analysis(t, x, nom_freq, tstart, half_bandwidth_frac)

    print(report_amp[1])
    print(report_t[1])
