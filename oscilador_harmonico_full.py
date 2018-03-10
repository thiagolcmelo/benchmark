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

# 
f = c / l # Hz
w = 2.0 * np.pi * f
x_m = np.linspace(-(L/2) * 1e-10, (L/2) * 1e-10, N)
v_j = 0.5 * me * x_m**2 * w**2
x_nm = x_m / 1e-9
v_ev = v_j / ev
v_au = v_ev / au2ev

# # runge-kutta ordem 4
# alpha = 1j / (2 * dx_au ** 2)
# beta = -1j * (v_au + 1.0 / (dx_au ** 2))
# diagonal_1 = beta #[beta] * N
# diagonal_2 = [alpha] * (N - 1)
# diagonais = [diagonal_1, diagonal_2, diagonal_2]
# D = diags(diagonais, [0, -1, 1]).toarray()
# def propagador(p):
#     k1 = D.dot(p)
#     k2 = D.dot(p + dt_au * k1 / 2)
#     k3 = D.dot(p + dt_au * k2 / 2)
#     k4 = D.dot(p + dt_au * k3)
#     return p + dt_au * (k1 + 2 * k2 + 2 * k3 + k4) / 6
# propagador_titulo = "Runge-Kutta Ordem 4"

# # crank-nicolson
# alpha = - dt_au * (1j / (2 * dx_au ** 2))/2.0
# beta = 1.0 - dt_au * (-1j * (v_au + 1.0 / (dx_au ** 2)))/2.0
# gamma = 1.0 + dt_au * (-1j * (v_au + 1.0 / (dx_au ** 2)))/2.0
# diagonal_1 = beta #[beta] * N
# diagonal_2 = [alpha] * (N - 1)
# diagonais = [diagonal_1, diagonal_2, diagonal_2]
# invB = inv(diags(diagonais, [0, -1, 1]).toarray())
# diagonal_3 = gamma #[gamma] * N
# diagonal_4 = [-alpha] * (N - 1)
# diagonais_2 = [diagonal_3, diagonal_4, diagonal_4]
# C = diags(diagonais_2, [0, -1, 1]).toarray()
# D = invB.dot(C)
# propagador = lambda p: D.dot(p)
# propagador_titulo = "Crank-Nicolson"

# split step
exp_v2 = np.exp(- 0.5j * v_au * dt_au)
exp_t = np.exp(- 0.5j * (2 * np.pi * k_au) ** 2 * dt_au)
propagador = lambda p: exp_v2 * ifft(exp_t * fft(exp_v2 * p))
propagador_titulo = "Split-Step"

# chutes iniciais
n = 6
short_grid = np.linspace(-1, 1, N)
g = gaussian(N, std=int(N/100))
estados = np.array([g * sp.legendre(i)(short_grid) for i in range(n)],dtype=np.complex_)
valores = np.zeros(n)
contadores = np.zeros(n)

valores_analiticos_si = [hbar_si * w * (i + 0.5) for i in range(n)]
valores_analiticos_ev = np.array(valores_analiticos_si) / ev

texto_x_l = -L/2
texto_x_r = 0.7 * L/2


for s in range(n):
    while True:
        estados[s] = propagador(estados[s])
        contadores[s] += 1
        
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
            p_h_p = simps(psi_conj * (-0.5 * derivada2 + v_au[1:-1] * psi), x_au[1:-1])
            # divide por <Psi|Psi> 
            p_h_p /= A
            valores[s] = p_h_p.real * au2ev # eV
            
#             # especificos do grafico
#             fig = plt.figure()
#             ax = fig.add_subplot(1, 1, 1)
#             ax.set_ylim([-0.1,1.1])
#             ax.spines["top"].set_visible(False)
#             ax.spines["right"].set_visible(False)
#             plt.title("Autoestados/Autovalores Oscilador Harmônico Quântico (%s)" % (propagador_titulo), fontsize=18)
#             plt.xlabel("x (\AA)", fontsize=16)
#             plt.ylabel(r'$E \, (eV)$', fontsize=16)
            
#             psif = [0.1 * estados[m]/np.ptp(estados[m]) + valores[m] for m in range(s+1)]
#             lines = []
#             for i, p in enumerate(psif):
#                 line, = plt.plot(x_au * au2ang, p, lw=1.0, color=tableau20[i], label=r'$|\Psi_{%d} (x,t)|^2$' % i)
#                 lines.append(line)
#                 ax.text(texto_x_l, valores[i] + 0.02, r"$E_{%d} = %.4f$ eV" % (i, valores[i]))
#                 ax.text(texto_x_r, valores[i] + 0.02, r"$%d$ K iterações" % int(contadores[i]/1000))
                
            
#             linev, = plt.plot(x_au * au2ang, v_au * au2ev, lw=1.0, color=tableau20[n], label='$V(x)$')
#             lines.append(linev)
#             plt.legend(handles=lines, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=4)
#             plt.show()
            
            print("%.4e / %.4e" % (valores[s], valores_analiticos_ev[s]))
            if np.abs(1-valores[s]/valores_analiticos_ev[s]) < 0.0001:
                break

