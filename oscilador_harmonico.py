#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Este módulo possui uma implementação pseudo-analítica e outra
numérica do oscilador harmônico quântico, é possível compará-las
em termos de quanto tempo o método numérico leva para atingir 
um determinado nível de precisão em relação à solução analítica

/ohq/i indica oscilador harmônico quântico
"""

# bibliotecas
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
from sklearn.preprocessing import StandardScaler

import os, time
from multiprocessing import Pool, TimeoutError
import logging

logger = logging.getLogger('onda_plana_logger')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(\
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# configuracao do matplotlib
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

# grandezas de interesse em unidades atomicas
au_l = cte.value('atomic unit of length')
au_t = cte.value('atomic unit of time')
au_e = cte.value('atomic unit of energy')

# outras relacoes de interesse
ev = cte.value('electron volt')
c = cte.value('speed of light in vacuum')
hbar_si = cte.value('Planck constant over 2 pi')
me = cte.value('electron mass')
au2ang = au_l / 1e-10
au2ev = au_e / ev

def _potencial_au(comp_onda, L, N):
    """
    Retorna um potencial e um grid espacial para um determinado
    comprimento de onda `comp_onda` em micrometros

    Parâmetros
    ----------
    comp_onda : float
        comprimento de onda em micrometros
    L : float
        tamanho do sistema em Angstrom
    N : int
        número de pontos para dividir o espaço

    Retorno
    -------
    potencial : tuple
        (`z_au`,`v_au`) um grid espacial em unidades atomicas e o 
        potencial em unidades atomicas
    """
    w = _omega(com_onda)
    z_m = np.linspace(-(L/2) * 1e-10, (L/2) * 1e-10, N)
    z_au = np.linspace(-L/au_l/2.0, L/au_l/2.0, N)
    v_j = 0.5 * me * z_m**2 * w**2 # potencial em Joules
    v_ev = v_j / ev # Joule para eV
    v_au = v_ev / au2ev # eV para au
    (z_au, v_au)

def _omega(com_onda):
    """
    Para um comprimento de onda em micrometros, retorna a frequencia
    angular em rad/seg

    Parâmetros
    ----------
    comp_onda : float
        o comprimento de onda em micrometros

    Retorno
    -------
    ang_freq : float
        frequencia angular em rad/seg
    """
    f = c / comp_onda # Hz
    return 2.0 * np.pi * f

def ohq_analitico(L=100.0, N=2048, comp_onda=8.1, nmax=6):
    """
    Esta função permite o cálculo dos `nmax` primeiros autoestados
    e autovalores do oscilador harmônico quântico para uma 
    frequência w=2*pi*c/lambda, onde lambda é o `comp_onda` em 
    micrometros

    Parâmetros
    ----------
    L : float
        tamanho do sistema em Angstrom
    N : int
        número de pontos para dividir o espaço
    comp_onda : float
        comprimento de onda em micrometros
    nmax : int
        número de autoestados/autovalores a serem calculados
    
    Retorno
    -------
    """

    z_au, v_au = _potencial_au(comp_onda, L, N)
    w = _omega(com_onda)

    # calculo do nmax autovalores
    autovalores_si = [hbar_si * w * (n+1/2) for n in range(nmax)]
    autovalores_ev = autovalores_si / ev

    # calculo dos nmax autoestados
    autoestados = []
    w_au = w * au_t
    mwoh_au = w_au # m * w / hbar em unidades atomicas
    for n in range(nmax):
        an = np.sqrt(1/(2**n * factorial(n)))*(mwoh_au/np.pi)**(1/4)
        psin = an*np.exp(-mwoh_au*z_au**2/2)*hermite(n)(z_au)
        autoestados.append(psin)


if __name__ == "__main__":
    res = ohq_analitico()