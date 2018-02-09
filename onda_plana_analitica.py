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
    
# unidades do problema
E_0 = 150.0 # eV
L = 400.0 # angstron
sigma = 5.0 # angstron
x_0 = -20.0 # angstron
dt = 1e-18 # s

# grandezas de interesse em unidades atomicas
au_l = cte.value('atomic unit of length')
au_t = cte.value('atomic unit of time')
au_e = cte.value('atomic unit of energy')

# outras relacoes de interesse
ev = cte.value('electron volt')
me = cte.value('electron mass')
hbar_ev = cte.value('Planck constant over 2 pi in eV s')
hbar = cte.value('Planck constant over 2 pi')
au2ang = au_l / 1e-10
au2ev = au_e / ev

# unidades atomicas
E_0_au = E_0 / au2ev
L_au = L / au2ang
sigma_au = sigma / au2ang
x_0_au = x_0 / au2ang
dt_au = dt / au_t
k_0_au = np.sqrt(2 * E_0_au)

N = 1024

# malha espacial
gx_au = np.linspace(-L_au/2, L_au/2, N)
x = np.linspace(-L/2, L/2, N)
dx_au = gx_au[1] - gx_au[0]
dx = x[1] - x[0]

# pacote de onda
p0_au = np.sqrt(2*E_0_au)
p0 = np.sqrt(2*me*(E_0*ev)/hbar**2)

z=lambda t_au: sigma_au**2 + 1.0j * t_au
psi_xt = lambda x_au, t_au: \
    np.sqrt(sigma_au/np.sqrt(np.pi)) * \
    (np.sqrt(z(t_au))/np.abs(z(t_au))**2) * \
    np.exp(1j*(p0_au*x_au-p0_au**2*t_au/2)) * \
    np.exp(1j*t_au*(x_au-p0_au*t_au/2)**2/(2*np.abs(z(t_au)**2))) * \
    np.exp(-sigma_au**2*(x_au-p0_au*t_au/2)**2/(2*np.abs(z(t_au)**2)))
psi_xt = np.vectorize(psi_xt)

psi = np.vectorize(psi_xt)(gx_au,0)
A0 = (simps(np.conjugate(psi)*psi,gx_au)).real
x_f_au = x_0_au

contador = 0
tempo = 0.0

n_label = r"N = %d" % N
dx_label = r"$\Delta x = %.2e$ \AA" % (L / N)
dt_label = r"$\Delta t = %.2e$ s" % (dt)

while x_f_au < -x_0_au:
    psi = psi_xt(gx_au,tempo / au_t)
    
    tempo += dt
    contador += 1
    if contador % 100 == 0:
        psis = np.conjugate(psi)
        A = (simps(psis*psi,gx_au)).real
        norma = 100 * A / A0
        x_f_au = xm = (simps(psis * gx_au * psi,gx_au)).real / A
        xm2 = (simps(psis * gx_au**2 * psi,gx_au)).real / A
        xm3 = (simps(psis * gx_au**3 * psi,gx_au)).real / A
        pm = (simps(-1j * hbar * psis[1:-1] * (psi[:-2] - 2 * psi[1:-1] + psi[2:]) / dx**2,x[1:-1])).real / (simps(psis*psi,x)).real
        
        sigma = np.sqrt(np.abs(xm2 - xm**2))
        gamma = (xm3 - 3*xm*sigma**2-xm**3)/sigma**3
        print("A = {:.5f} %, <x> = {:.5f} A, sigma = {:.5f}".format(norma, xm, sigma))
        # especificos do grafico
        ax = plt.subplot(111)
        ax.set_ylim([-0.05,0.5])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(-L/2, 0.5, r"$A = %.2f$ \%%" % (norma))
        ax.text(-L/2, 0.45, r"$p_0 = %.2e$" % (p0))
        ax.text(-L/2, 0.4, r"$\langle p \rangle = %.2e$" % (pm))
        ax.text(-L/2, 0.35, r"$\langle x \rangle = %.2f$ \AA" % (xm *au2ang))
        ax.text(-L/2, 0.30, r"$3 \sigma = %.2f$ \AA" % (3*sigma *au2ang))
        ax.text(-L/2, 0.25, r"$\gamma = %.2f$" % (gamma))
        ax.text(-L/2, 0.20, r"$t = %.2e$ s" % (tempo))
        ax.text(-L/2, 0.15, n_label)
        ax.text(-L/2, 0.10, dx_label)
        ax.text(-L/2, 0.05, dt_label)
        plt.title("Pacote de Onda Plana (Evolução Analitica)", fontsize=18)
        plt.xlabel("x (\AA)", fontsize=16)
        plt.ylabel(r'$|\Psi (x,t)|^2$', fontsize=16)
        line, = plt.plot(gx_au * au2ang, np.abs(psi), lw=2.0, color=tableau20[0], label='Inicial')
        plt.show()