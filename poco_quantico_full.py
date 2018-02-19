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
L = 200 # angstron
N = 4096
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
v_au = np.vectorize(lambda x: Vb_au if np.abs(x) > a_au/2 else 0.0)(x_au)
me_eff = np.vectorize(lambda x: me_algaas if np.abs(x) > a_au/2 else me_gaas)(x_au)

# # runge-kutta ordem 4
# alpha = 1j / (2 * me_eff * dx_au ** 2)
# beta = -1j * (v_au + 1.0 / (me_eff * dx_au ** 2))
# diagonal_1 = beta
# diagonal_2 = alpha[1:]
# diagonal_3 = alpha[:-1]
# diagonais = [diagonal_1, diagonal_2, diagonal_3]
# D = diags(diagonais, [0, 1, -1]).toarray()
# def propagador(p):
#     k1 = D.dot(p)
#     k2 = D.dot(p + dt_au * k1 / 2)
#     k3 = D.dot(p + dt_au * k2 / 2)
#     k4 = D.dot(p + dt_au * k3)
#     return p + dt_au * (k1 + 2 * k2 + 2 * k3 + k4) / 6
# propagador_titulo = "Runge-Kutta Ordem 4"

# crank-nicolson
alpha = - dt_au * (1j / (2 * me_eff * dx_au ** 2))/2.0
beta = 1.0 - dt_au * (-1j * (v_au + 1.0 / (me_eff * dx_au ** 2)))/2.0
gamma = 1.0 + dt_au * (-1j * (v_au + 1.0 / (me_eff * dx_au ** 2)))/2.0
diagonal_1 = beta
diagonal_2_1 = alpha[1:]
diagonal_2_2 = alpha[:-1]
diagonais = [diagonal_1, diagonal_2_1, diagonal_2_2]
invB = inv(diags(diagonais, [0, 1, -1]).toarray())
diagonal_3 = gamma
diagonal_4_1 = -alpha[1:]
diagonal_4_2 = -alpha[:-1]
diagonais_2 = [diagonal_3, diagonal_4_1, diagonal_4_2]
C = diags(diagonais_2, [0, 1, -1]).toarray()
D = invB.dot(C)
propagador = lambda p: D.dot(p)
propagador_titulo = "Crank-Nicolson"

# # split step
# me_eff = np.ones(N) * me_gaas
# exp_v2 = np.exp(- 0.5j * v_au * dt_au)
# exp_t = np.exp(- 0.5j * (2 * np.pi * k_au) ** 2 * dt_au / me_eff)
# propagador = lambda p: exp_v2 * ifft(exp_t * fft(exp_v2 * p))
# propagador_titulo = "Split-Step"

# chutes iniciais
n = 6
short_grid = np.linspace(-1, 1, N)
g = gaussian(N, std=int(N/50))
estados = np.array([g * sp.legendre(i)(short_grid) for i in range(n)],dtype=np.complex_)
valores = np.zeros(n)
contadores = np.zeros(n)

valores_analiticos_ev = [0.044280126, 0.176480128, 0.394408742, 0.693159988, 1.060713269, 1.438011481]

texto_x_l = -L/2
texto_x_r = 0.7 * L/2

for s in range(n):
    v_ant = 1.0
    while True:
        contadores[s] += 1
        
        estados[s] = propagador(estados[s])
        
        # gram-shimdt
        for m in range(s):
            proj = simps(estados[s] * np.conjugate(estados[m]), x_au)
            estados[s] -= proj * estados[m]
            
        # normaliza
        A = np.sqrt(simps(np.abs(estados[s])**2, x_au))
        estados[s] /= np.sqrt(simps(np.abs(estados[s])**2, x_au))
        
        if contadores[s] % 1000 == 0:
            # calcula autoestados
            # derivada segunda
            derivada2 = (estados[s][:-2] - 2 * estados[s][1:-1] + estados[s][2:]) / dx_au**2
            psi = estados[s][1:-1]
            psi_conj = np.conjugate(psi)
            # <Psi|H|Psi>
            p_h_p = simps(psi_conj * (-0.5 * derivada2 / me_eff[1:-1] + v_au[1:-1] * psi), x_au[1:-1])
            # divide por <Psi|Psi> 
            p_h_p /= A
            valores[s] = p_h_p.real * au2ev # eV
            
            # especificos do grafico
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_ylim([-0.1,1.5])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            #plt.title("Autoestados/Autovalores Poço Quântico (%s)" % (propagador_titulo), fontsize=18)
            plt.xlabel("x (\AA)", fontsize=16)
            plt.ylabel(r'$E \, (eV)$', fontsize=16)
            
            psif = [np.abs(estados[m]) for m in range(s+1)]
            psif = [0.1 * p / np.ptp(p) + valores[m] for m, p in enumerate(psif)]
            lines = []
            for i, p in enumerate(psif):
                line, = plt.plot(x_au * au2ang, p, lw=1.0, color=tableau20[i], label=r'$|\Psi_{%d} (x,t)|^2$' % i)
                lines.append(line)
                ax.text(texto_x_l, valores[i] + 0.02, r"$E = %.4f$ eV" % (valores[i]))
                ax.text(texto_x_r, valores[i] + 0.02, r"$Iter = %d$" % (contadores[i]))
                
            
            linev, = plt.plot(x_au * au2ang, v_au * au2ev, lw=1.0, color=tableau20[n], label='$V(x)$')
            lines.append(linev)
            plt.legend(handles=lines, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=4)
            plt.show()
            
            print("%d >>> %.4e / %.4e" % (contadores[s], valores[s], valores_analiticos_ev[s]))
            
            #if np.abs(1-valores[s]/valores_analiticos_ev[s]) < 0.001:
            if np.abs(1-valores[s]/v_ant) < 0.0001:
                break
            else:
                v_ant = valores[s]