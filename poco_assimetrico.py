# libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
import scipy.constants as cte
from scipy.sparse import diags
from scipy.linalg import inv
from scipy.fftpack import fft, ifft, fftfreq
import scipy.special as sp
from scipy.signal import gaussian

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
L = 0.825*512 # angstron
N = 512
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
me_x = lambda x: 0.067+0.083*x
algaas_x = 0.4
Eg = eg(algaas_x)
me_algaas = me_x(algaas_x)
me_gaas = me_x(0.0)
Vb_au = Eg / au2ev
a = 100 # angstron
a_au = a / au2ang

adw_k0 = -132.7074997
k2 = 7.0
k3 = 0.5
k4 = 1.0
v_au = np.vectorize(lambda x: adw_k0-k2*x**2+k3*x**3+k4*x**4)(x_au)
me_eff = np.vectorize(lambda x: me_algaas if np.abs(x) > a_au/2 else me_gaas)(x_au)


# especificos do grafico
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#ax.set_ylim([-5000,100])
ax.set_ylim([-150,-100])
ax.set_xlim([-6, 6])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
#plt.title("Autoestados/Autovalores Poço Quântico (%s)" % (propagador_titulo), fontsize=18)
plt.xlabel("x (\AA)", fontsize=16)
plt.ylabel(r'$E \, (eV)$', fontsize=16)

lines = []
linev, = plt.plot(x_au, v_au, lw=1.0, color=tableau20[0], label='$V(x)$')
lines.append(linev)
plt.legend(handles=lines, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=4)
plt.show()