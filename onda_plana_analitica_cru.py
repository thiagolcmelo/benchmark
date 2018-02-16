# libraries
import warnings
warnings.filterwarnings('ignore')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
import scipy.constants as cte
from scipy.sparse import diags
from scipy.linalg import inv
from scipy.stats import skew
from scipy.stats import describe


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

# outras relacoes de interesse
ev = cte.value('electron volt')
me = cte.value('electron mass')
hbar = cte.value('Planck constant over 2 pi')

# unidades do problema
E_0 = 150.0 * ev # eV > J
L = 400.0 * 1e-10 # angstron
delta_x = 10.0 * 1e-10 # angstron
x_0 = -20.0 * 1e-10 # angstron
k_0 = np.sqrt(2 * me * E_0 / hbar**2)
p_0 = np.sqrt(2 * me * E_0)
dt = 1e-18 # s
N = 1024

# malha espacial
grid_x = np.linspace(-L/2, L/2, N)
dx = grid_x[1] - grid_x[0]

# pacote de onda
z=lambda t: delta_x**2 + 1.0j * hbar * t / me
psi_xt = lambda x, t: \
    np.sqrt(delta_x/np.sqrt(np.pi)) * \
    (np.sqrt(z(t))/np.abs(z(t))**2) * \
    np.exp(1j*(p_0*x-p_0**2*t/(2*me))/hbar) * \
    np.exp(1j*hbar*t*(x-p_0*t/(2*me))**2/(2*me*np.abs(z(t)**2))) * \
    np.exp(-delta_x**2*(x-p_0*t/(2*me))**2/(2*np.abs(z(t)**2)))
psi_xt = np.vectorize(psi_xt)

psi = np.vectorize(psi_xt)(grid_x,0)
A0 = (simps(np.abs(psi)**2,grid_x))
x_f = x_0

contador = 0
tempo = 0.0

n_label = r"N = %d" % N
dx_label = r"$\Delta x = %.2e$ \AA" % (L / N / 1e-10)
dt_label = r"$\Delta t = %.2e$ s" % (dt)

while x_f < L/4:
    psi = psi_xt(grid_x,tempo)
    tempo += dt
    contador += 1
    if contador % 100 == 0:
        psis = np.conjugate(psi)
        A = np.abs(simps(np.abs(psi)**2,grid_x))
        Av = np.abs(simps(np.abs(psi)**2,grid_x / 1e-9))
        
        psif = np.abs(psi)**2 / Av
        
        norma = 100 * A / A0
        x_f = xm1 = np.abs(simps(psis * grid_x * psi,grid_x)) / A
        xm2 = np.abs(simps(psis * grid_x**2 * psi,grid_x)) / A
        xm3 = np.abs(simps(psis * grid_x**3 * psi,grid_x)) / A
        pm = np.abs(simps(-1j * hbar * psis[1:-1] * (psi[2:]-psi[:-2]) / (2*dx),grid_x[1:-1])) / np.abs(simps(psis[1:-1]*psi[1:-1],grid_x[1:-1]))
        sigma = np.sqrt(np.abs(xm2 - xm1**2))
        gamma = (xm3 - 3*xm1*sigma**2-xm1**3)/sigma**3
        
        print("A = {:.5f} %, <x> = {:.5f} A, sigma = {:.5f}".format(norma, xm1 / 1e-10, sigma / 1e-10))
        # especificos do grafico
        ax = plt.subplot(111)
        #ax.set_ylim([-0.05,0.5])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(-L/2e-10, 0.5, r"$A = %.2f$ \%%" % (norma))
        ax.text(-L/2e-10, 0.45, r"$p_0 = %.2e$" % (p_0))
        ax.text(-L/2e-10, 0.4, r"$\langle p \rangle = %.2e$" % (pm))
        ax.text(-L/2e-10, 0.35, r"$\langle x \rangle = %.2f$ \AA" % (xm1 / 1e-10))
        ax.text(-L/2e-10, 0.30, r"$3 \sigma = %.2f$ \AA" % (3*sigma / 1e-10))
        ax.text(-L/2e-10, 0.25, r"$\gamma = %.2f$" % (gamma))
        ax.text(-L/2e-10, 0.20, r"$t = %.2e$ s" % (tempo))
        ax.text(-L/2e-10, 0.15, n_label)
        ax.text(-L/2e-10, 0.10, dx_label)
        ax.text(-L/2e-10, 0.05, dt_label)
        plt.title("Pacote de Onda Plana (Evolução Analitica)", fontsize=18)
        plt.xlabel("x (\AA)", fontsize=16)
        plt.ylabel(r'$|\Psi (x,t)|^2$', fontsize=16)
        line, = plt.plot(grid_x/1e-10, psif, lw=2.0, color=tableau20[0], label='Inicial')
        plt.show()








