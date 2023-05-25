# coding: utf-8
%run ./load_M_K.py
eigvecs[:, 0].reshape(-1,6)
approx_flap = np.zeros(M.shape[0])
approx_flap = np.zeros(Mfull.shape[0])
approx_flap[::6] = 1.0
eigvecs.shape
approx_flap @ eigvecs
flapind = np.argmax(np.abs(approx_flap @ eigvecs))
flapind
flapind = np.argmax(np.abs(approx_flap @ eigvecs[:, :9]))
flapind
approx_edge = np.zeros(Mfull.shape[0])
approx_edge[1::6] = 1.0
approx_edge @ eigvecs
approx_twist = np.zeros(Mfull.shape[0]))
approx_twist = np.zeros(Mfull.shape[0])
approx_twist[5::6] = 1.0
approx_twist @ eigvecs
ls
%save -r mode_id_ipython.py 1-99999
