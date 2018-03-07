#!/usr/bin/env python
"""
This module just solves the analytical expression for a
very specific case of quantum well, which is one made
of GaAs surrounded by AlGaAs with concentration x=0.4
"""

# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import simps
import scipy.constants as cte
from scipy.sparse import diags
from scipy.linalg import inv
from scipy.fftpack import fft, ifft, fftfreq
import scipy.special as sp
from scipy.signal import gaussian
from scipy.optimize import newton, fsolve

# matplotlib defaults setup
plt.rcParams['savefig.dpi'] = 75
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 14, 8
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "computer modern sans serif"
plt.rcParams['text.usetex']=True
plt.rcParams['text.latex.unicode']=True

# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

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

# constantes do problema
E0 = 150.0 # eV
delta_x = 5.0 # angstron
x0 = -30.0 # angstron
xf = -40.0 # angstron
l = 8.1e-6 # m

# otimizando
L = 100 # angstron
N = 2048
hN = int(N/2)
dt = 1e-19 # s

# unidades atomicas
L_au = L / au2ang
dt_au = -1j * dt / au_t
E0_au = E0 / au2ev
delta_x_au = delta_x / au2ang
x0_au = x0 / au2ang
xf_au = xf / au2ang
k0_au = np.sqrt(2 * E0_au)

# malhas direta e reciproca
dx = L / (N-1)
x_au = np.linspace(-L_au/2.0, L_au/2.0, N)
dx_au = np.abs(x_au[1] - x_au[0])
k_au = fftfreq(N, d=dx_au)

# props do material
eg = lambda x: 0.7 * (1.519 + 1.447 * x - 0.15 * x**2) # eV
Vb_au = (eg(0.4)-eg(0.0)) / au2ev
me_eff = 0.067

valores = []
#poco_a = np.linspace(10,500,200)
poco_a = [100]
series = []
for a in poco_a:
    autovalores = []
    
    a_au = a / au2ang
    
    v_au = np.vectorize(lambda x: Vb_au if np.abs(x) > a_au/2 else 0.0)(x_au)
    
    f = lambda e: np.tan(np.sqrt(2*me_eff*e)*a_au/2)-np.sqrt(2*me_eff*(Vb_au-e)) / np.sqrt(2*me_eff*e)
    
    for e0 in np.linspace(-Vb_au, Vb_au, 1000):
        try:
            root = newton(f, x0=e0)
            if root > 0:
                autovalores.append(root * au2ev)
        except:
            pass
    
    f = lambda e: 1.0/np.tan(np.sqrt(2*me_eff*e)*a_au/2)+np.sqrt(2*me_eff*(Vb_au-e)) / np.sqrt(2*me_eff*e)
    
    for e0 in np.linspace(-Vb_au, Vb_au, 1000):
        try:
            root = newton(f, x0=e0)
            if root > 0:
                autovalores.append(root * au2ev)
        except:
            pass
        
    autovalores = list(sorted(set(autovalores)))
    print(autovalores)
    for i, av in enumerate(autovalores):
        series.append((a, av))

#pd.DataFrame(series, columns=['a', 'E']).to_csv('analytic_quantum_well.csv')
        
# especificos do grafico
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# plt.title("Autoestados de um Poço Quântico ($V_b=%.3f$ eV)" % (Vb_au * au2ev), fontsize=22)
# plt.xlabel("a (\AA)", fontsize=20)
# plt.ylabel(r'$E \, (eV)$', fontsize=20)
# lines = []
# x = [i for i,_ in series]
# y = [j for _,j in series]
# plt.scatter(x, y, c=tableau20[0], label='$E_{%d}$' % 0, s=2)
# plt.show()
