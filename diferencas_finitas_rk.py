# libraries
import warnings
warnings.filterwarnings('ignore')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
import scipy.constants as cte
from scipy.sparse import diags


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
au2ang = au_l / 1e-10
au2ev = au_e / ev

# unidades do problema
E_0 = 150.0 # eV
L = 100.0 # angstron
sigma_x = 1.0 # angstron
x_0 = -20.0 # angstron
dt = dt_0 = 1e-19 # s

# unidades atomicas
E_0_au = E_0 / au2ev
L_au = L / au2ang
sigma_x_au = sigma_x / au2ang
x_0_au = x_0 / au2ang
dt_au = dt / au_t
k_0_au = np.sqrt(2 * E_0_au)

N = 2048
#dt = 1e-19

# malha espacial
x_au = np.linspace(-L_au/2, L_au/2, N)
dx_au = x_au[1] - x_au[0]

# diferencas finitas
alpha = 1j / (2 * dx_au ** 2)
beta = - 1j / (dx_au ** 2)
diagonal_1 = [beta] * N
diagonal_2 = [alpha] * (N - 1)
diagonais = [diagonal_1, diagonal_2, diagonal_2]
M = diags(diagonais, [0, -1, 1]).toarray()

# pacote de onda
PN = 1/(2*np.pi*sigma_x_au**2)**(1/4)
psi = PN*np.exp(1j*k_0_au*x_au-(x_au-x_0_au)**2/(4*sigma_x_au**2))
A0 = (simps(np.conjugate(psi)*psi,x_au)).real
x_f_au = x_0_au

contador = 0
tempo = 0.0

n_label = r"N = %d" % N
dx_label = r"$\Delta x = %.2e$ \AA" % (L / N)
dt_label = r"$\Delta t = %.2e$ s" % (dt)

while x_f_au < -x_0_au:
    k1 = M.dot(psi)
    k2 = M.dot(psi + dt_au * k1 / 2)
    k3 = M.dot(psi + dt_au * k2 / 2)
    k4 = M.dot(psi + dt_au * k3)
    psi += dt_au * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    tempo += dt
    contador += 1
    if contador % 100 == 0:    
        norma = 100 * A / A0
        A = (simps(np.conjugate(psi)*psi,x_au)).real
        
        x_f_au = xm = (simps(np.conjugate(psi)* x_au * psi,x_au)).real / A
        xm2 = (simps(np.conjugate(psi)* x_au**2 * psi,x_au)).real / A
        xm3 = (simps(np.conjugate(psi)* x_au**3 * psi,x_au)).real / A
        
        sigma = np.sqrt(np.abs(xm2 - xm**2))
        gamma = (xm3 - 3*xm*sigma**2-xm**3)/sigma**3
        
        print("A = {:.5f} %, <x> = {:.5f} A, sigma = {:.5f}".format(norma, xm, sigma))
        
        # especificos do grafico
        ax = plt.subplot(111)
        ax.set_ylim([-0.05,0.5])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        ax.text(-50.0, 0.4, r"$A = %.2f$ \%%" % (norma))
        ax.text(-50.0, 0.35, r"$\langle x \rangle = %.2f$ \AA" % (xm *au2ang))
        ax.text(-50.0, 0.30, r"$3 \sigma = %.2f$ \AA" % (3*sigma *au2ang))
        ax.text(-50.0, 0.25, r"$\gamma = %.2f$" % (gamma))
        ax.text(-50.0, 0.20, r"$t = %.2e$ s" % (tempo))
        ax.text(-50.0, 0.15, n_label)
        ax.text(-50.0, 0.10, dx_label)
        ax.text(-50.0, 0.05, dt_label)
        
        plt.title("Pacote de Onda Plana (Evolução Temporal RK ordem 4)", fontsize=18)
        plt.xlabel("x (\AA)", fontsize=16)
        plt.ylabel(r'$|\Psi (x,t)|^2$', fontsize=16)
        
        line, = plt.plot(x_au * au2ang, np.abs(psi), lw=2.0, color=tableau20[0], label='Inicial')
        #plt.legend(handles=[line], loc=1)
        #plt.legend()
        plt.show()



