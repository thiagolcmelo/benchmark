#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module implements an analytical along side with a numerical
solution for the quantum harmonic oscillator eigenstates and 
eigenvalues. It is possible to compare them and plot a chart about
how long it takes for achieving some arbitrary level of precision
using the numerical method.

/qho/i stands for qho
/AU/ stands for Atomic Units
"""

# libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial
from scipy.special import hermite
from scipy.integrate import simps
import scipy.constants as cte
from scipy.sparse import diags
from scipy.linalg import inv
from scipy.fftpack import fft, ifft, fftfreq
from scipy.spatial.distance import cdist
from scipy.signal import gaussian
import scipy.special as sp
from sklearn.preprocessing import StandardScaler
import os, time
from multiprocessing import Pool, TimeoutError
import logging

# setiing up a logger for clear program messages
logger = logging.getLogger('qho-logger')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(\
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# setting up matplotlib default config
plt.rcParams['savefig.dpi'] = 75
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 14, 8
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "computer modern sans serif"
plt.rcParams['text.usetex']=True
plt.rcParams['text.latex.unicode']=True

# AU of interest
AU_L = cte.value('atomic unit of length')
AU_T = cte.value('atomic unit of time')
AU_E = cte.value('atomic unit of energy')

# other units, relations, and constants of interest
ev = cte.value('electron volt')
c = cte.value('speed of light in vacuum')
hbar_si = cte.value('Planck constant over 2 pi')
me = cte.value('electron mass')
au2ang = AU_L / 1e-10
au2ev = AU_E / ev

def _potential_au(wave_length, L, N):
    """
    For a given `wave_length` (enclosed in a space of lenth `L` and
    represented by `N` points), this function returns the associated
    quantum harmonic oscillator potential. It sets the origin at
    the middle of [-L/2,+L/2].

    Parameters
    ----------
    wave_length : float
        the wave length in meters
    L : float
        the length of the space in Angstrom
    N : int
        the number of points in the space

    Return
    ------
    potential : tuple
        (`z_au`,`v_au`) where `z_au` stands for the spatial grid in
        AU and `v_au` stands for the associated potencial
        also in AU
    """
    w, _ = _omega(wave_length)
    z_si = np.linspace(-(L/2) * 1e-10, (L/2) * 1e-10, N)
    z_au = np.linspace(-L/au2ang/2.0, L/au2ang/2.0, N)
    v_si = 0.5 * me * z_si**2 * w**2 # potential in Joules
    v_ev = v_si / ev # Joules to eV
    v_au = v_ev / au2ev # eV to au
    return z_si, z_au, v_ev, v_si, v_au

def _omega(wave_length):
    """
    For a given wave length in metters, returns the angular
    frequency in rad/sec

    Parameters
    ----------
    wave_length : float
        the wave length in meters

    Returns
    -------
    ang_freq : tuple
        angular frequency in rad/sec as (ang_freq_si, ang_freq_au)
    """
    f = c / wave_length # Hz
    w = 2.0 * np.pi * f
    return w, w * AU_T

def qho_numerical(L=100.0, N=2048, dt=1e-19, \
    wave_length=8.1e-6, nmax=6, precision=1e-2, \
    iterations=None, max_time=None, eigenstates_au=None):
    """
    This function calculates the first `nmax` eigenvalues and 
    eigenstates for a quantum harmonic oscillater corresponding to a
    given `wave_length`.

    Parameters
    ----------
    L : float
        the length of the space in Angstrom
    N : int
        the number of points in the space
    dt : float
        the time step in seconds
    wave_length : float
        the wave length in meters
    nmax : int
        number of wigenvalues/eigenstates for being calculated
    precision : float
        the eigenvalues minimum precision in percentage
    iterations : int
        a fixed number of iterations to use instead of a minimum
        precision
    max_time : float
        max time for wait on each eigenvalue/eigenstate to use instead
        of precision
    eigenstates_au : array_like
        an array of kickstart eigenstates
    
    Returns
    -------
    result : dictionary
        A dictionary with the following keys:
        - `z_si` the spatial grid in meters
        - `z_au` the spatial grid in atomig units
        - `v_au` the potential in AU
        - `v_ev` the potential in eV
        - `v_si` the potential in SI
        - `eigenvalues_si` the eigenvalues in Joules
        - `eigenvalues_ev` the eigenvalues in eV
        - `eigenvalues_au` the eigenvalues in AU
        - `eigenstates_au` the eigenstates in AU
        - `eigenstates_2_au` the eigenstates in the shape of |psi|^2
        - `eigenstates_si` the eigenstates in SI
        - `eigenstates_2_si` the eigenstates in the shape of |psi|^2
        - `iterations` an array with the number of iterations for 
            achieve precision on each state
        - `timers` an array with the number of seconds each
            eigenvalue/eigenstate pair took for being evolved
        - `precisions` an array with the precision of each eigenvalue
        - `chebyshev` chebyshev total distance of each eigenstate
        - `seuclidean` euclidean total distance of each eigenstate
        - `sqeuclidean` squared euclidean total distance of each 
            eigenstate
    """
    assert L > 0 # length most not be null
    assert wave_length > 0 # wave length most not be null
    assert nmax > 0 # must calculate at least one eigenvalue/eigenstate
    assert int(np.log2(N)) == np.log2(N) # must be a power of 2

    # get analytical solutions
    analytical = qho_analytical(L=L, N=N, \
        wave_length=wave_length, nmax=nmax)
    eigenvalues_ev_ana = analytical['eigenvalues_ev']
    eigenstates_au_ana = analytical['eigenstates_au']

    # grid values
    z_si, z_au, v_ev, v_si, v_au = _potential_au(wave_length, L, N)
    dt_au = -1j * dt / AU_T
    precision /= 100 # it is a percentage

    # split step
    dz_au = np.abs(z_au[1] - z_au[0])
    k_au = fftfreq(N, d=dz_au)
    exp_v2 = np.exp(- 0.5j * v_au * dt_au)
    exp_t = np.exp(- 0.5j * (2 * np.pi * k_au) ** 2 * dt_au)
    evolution_operator = lambda p: exp_v2*ifft(exp_t*fft(exp_v2*p))
    
    # chutes iniciais
    if not eigenstates_au:
        short_grid = np.linspace(-1, 1, N)
        g = gaussian(N, std=int(N/100))
        eigenstates_au = np.array([g*sp.legendre(i)(short_grid) \
            for i in range(nmax)],dtype=np.complex_)
        eigenvalues_ev = np.zeros(nmax)
    counters = np.zeros(nmax)
    timers = np.zeros(nmax)
    precisions = np.zeros(nmax)
    vectors_chebyshev = np.zeros(nmax)
    vectors_sqeuclidean = np.zeros(nmax)
    vectors_seuclidean = np.zeros(nmax)

    for s in range(nmax):
        while True:
            start_time = time.time()
            eigenstates_au[s] = evolution_operator(eigenstates_au[s])
            counters[s] += 1
            
            # gram-shimdt
            for m in range(s):
                proj = simps(eigenstates_au[s] * \
                    np.conjugate(eigenstates_au[m]), z_au)
                eigenstates_au[s] -= proj * eigenstates_au[m]
                
            # normalize
            A = np.sqrt(simps(np.abs(eigenstates_au[s])**2, z_au))
            eigenstates_au[s] /= A
            timers[s] += time.time() - start_time
            
            if (iterations and counters[s] >= iterations) \
                or (max_time and timers[s] >= max_time) \
                or counters[s] % 1000 == 0:
                # second derivative
                derivative2 = (eigenstates_au[s][:-2] - 2 * eigenstates_au[s][1:-1] + eigenstates_au[s][2:]) / dz_au**2
                psi = eigenstates_au[s][1:-1]
                psi_conj = np.conjugate(psi)
                # <Psi|H|Psi>
                p_h_p = simps(psi_conj * (-0.5 * derivative2 + v_au[1:-1] * psi), z_au[1:-1])
                # divide por <Psi|Psi> 
                p_h_p /= A**2
                eigenvalues_ev[s] = p_h_p.real * au2ev # eV
                
                precisions[s] = np.abs(1-eigenvalues_ev[s] \
                    / eigenvalues_ev_ana[s])
                
                if (iterations and counters[s] >= iterations) \
                    or (max_time and timers[s] >= max_time) \
                    or (not iterations and not max_time \
                        and precisions[s] < precision):
                    XA = [eigenstates_au[s]]
                    XB = [eigenstates_au_ana[s]]
                    vectors_chebyshev[s] = \
                        cdist(XA, XB, 'chebyshev')[0][0]
                    vectors_seuclidean[s] = \
                        cdist(XA, XB, 'seuclidean')[0][0]
                    vectors_sqeuclidean[s] = \
                        cdist(XA, XB, 'sqeuclidean')[0][0]
                    break
    
    # generate eigenstates for SI
    eigenstates_si = np.array([np.ones(N, dtype=np.complex_) \
        for i in range(nmax)],dtype=np.complex_)
    for i, state in enumerate(eigenstates_au):
        A_si = np.sqrt(simps(np.abs(state)**2, z_si))
        eigenstates_si[i] = state / A_si

    return {
        'z_si': z_si,
        'z_au': z_au,
        'v_au': v_au,
        'v_ev': v_ev,
        'v_si': v_si,
        'eigenvalues_si': eigenvalues_ev * ev,
        'eigenvalues_ev': eigenvalues_ev,
        'eigenvalues_au': eigenvalues_ev / au2ev,
        'eigenstates_au': eigenstates_au,
        'eigenstates_2_au': np.abs(eigenstates_au)**2,
        'eigenstates_si': eigenstates_si,
        'eigenstates_2_si': np.abs(eigenstates_si)**2,
        'iterations': counters,
        'timers': timers,
        'precisions': precisions,
        'chebyshev': vectors_chebyshev,
        'seuclidean': vectors_seuclidean,
        'sqeuclidean': vectors_sqeuclidean
    }

def qho_analytical(L=100.0, N=2048, wave_length=8.1e-6, nmax=6):
    """
    This function calculates the first `nmax` eigenvalues and 
    eigenstates for a quantum harmonic oscillater corresponding to a
    given `wave_length`.

    Parameters
    ----------
    L : float
        the length of the space in Angstrom
    N : int
        the number of points in the space
    wave_length : float
        the wave length in meters
    nmax : int
        number of wigenvalues/eigenstates for being calculated
    
    Returns
    -------
    result : dictionary
        A dictionary with the following keys:
        - `z_si` the spatial grid in meters
        - `z_au` the spatial grid in atomig units
        - `v_au` the potential in AU
        - `v_ev` the potential in eV
        - `v_si` the potential in SI
        - `eigenvalues_si` the eigenvalues in Joules
        - `eigenvalues_ev` the eigenvalues in eV
        - `eigenvalues_au` the eigenvalues in AU
        - `eigenstates_au` the eigenstates in AU
        - `eigenstates_2_au` the eigenstates in the shape of |psi|^2
        - `eigenstates_si` the eigenstates in SI
        - `eigenstates_2_si` the eigenstates in the shape of |psi|^2
    """
    assert L > 0 # length most not be null
    assert wave_length > 0 # wave length most not be null
    assert nmax > 0 # must calculate at least one eigenvalue/eigenstate
    assert int(np.log2(N)) == np.log2(N) # must be a power of 2

    # grid values
    z_si, z_au, v_ev, v_si, v_au = _potential_au(wave_length, L, N)
    w, w_au = _omega(wave_length)

    # nmax eigenvalues
    eigenvalues_si = [hbar_si * w * (n+1/2) for n in range(nmax)]
    eigenvalues_si = np.array(eigenvalues_si)
    eigenvalues_ev = eigenvalues_si / ev

    # nmax eigenstates
    eigenstates_si = []
    eigenstates_au = []
    mwoh_au = w_au # m * w / hbar in AU
    mwoh_si = me * w / hbar_si # m * w / hbar in si units
    for n in range(nmax):
        an_au = np.sqrt(1.0/(2.0**n * factorial(n))) * \
            (mwoh_au/np.pi)**(1.0/4.0)
        psin_au = an_au*np.exp(-mwoh_au*z_au**2/2.0) * \
            hermite(n)(np.sqrt(mwoh_au)*z_au)
        eigenstates_au.append(psin_au)

        an_si = np.sqrt(1.0/(2.0**n * factorial(n))) * \
            (mwoh_si/np.pi)**(1.0/4.0)
        psin_si = an_si*np.exp(-mwoh_si*z_si**2/2.0) * \
            hermite(n)(np.sqrt(mwoh_si)*z_si)
        eigenstates_si.append(psin_si)

    return {
        'z_si': z_si,
        'z_au': z_au,
        'v_au': v_au,
        'v_ev': v_ev,
        'v_si': v_si,
        'eigenvalues_si': eigenvalues_si,
        'eigenvalues_ev': eigenvalues_ev,
        'eigenvalues_au': eigenvalues_ev / au2ev,
        'eigenstates_au': eigenstates_au,
        'eigenstates_2_au': np.abs(eigenstates_au)**2,
        'eigenstates_si': eigenstates_si,
        'eigenstates_2_si': np.abs(eigenstates_si)**2,
    }

def comparison(n=6):

    #res_analytic = qho_analytical(nmax=n)
    
    res_numeric_iter = []
    for it in [i * 1e5 for i in range(1,11)]:
        res_numeric_iter.append(qho_numerical(nmax=n, iterations=it))
    
    res_numeric_time = []
    for t in [60 * i for i in range(1,6)]:
        res_numeric_time.append(qho_numerical(nmax=n, max_time=t))

    res_numeric_prec = []
    for p in [1/10**i for i in range(2,6)]:
        res_numeric_prec.append(qho_numerical(nmax=n, precision=p))

    # now what?
    
if __name__ == "__main__":
    resa = qho_analytical(nmax=1)
    resn = qho_numerical(nmax=1)
    for n in range(len(resa['eigenstates_si'])):
        plt.scatter(resa['z_si'], resa['eigenstates_si'][n], label='analytic')
        plt.scatter(resn['z_si'], resn['eigenstates_si'][n], label='numeric')
    plt.legend()
    plt.show()