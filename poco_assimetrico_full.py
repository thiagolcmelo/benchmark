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

# otimizando
L = 10.0 # angstron
N = 2048
hN = int(N/2)
dt = 1e-20 # s

# unidades atomicas
L_au = L / au2ang
dt_au = -1j * dt / au_t

# malhas direta e reciproca
x_au = np.linspace(-L_au/2.0, L_au/2.0, N)
dx_au = np.abs(x_au[1] - x_au[0])
k_au = fftfreq(N, d=dx_au)

# props do material
me_eff = 0.5
adw_k0 = 0.0#-132.7074997
k2 = 7.0
k3 = 0.5
k4 = 1.0
v_adw = lambda x: adw_k0-k2*x**2+k3*x**3+k4*x**4
v_au = np.vectorize(v_adw)(x_au)

# split step
exp_v2 = np.exp(- 0.5j * v_au * dt_au)
exp_t = np.exp(- 0.5j * (2 * np.pi * k_au) ** 2 * dt_au / me_eff)
propagador = lambda p: exp_v2 * ifft(exp_t * fft(exp_v2 * p))
propagador_titulo = "Split-Step"

# chutes iniciais
n = 9
a = 1.9
sigma = 0.87
g = np.vectorize(lambda x: np.exp(-(x-a)**2/(2*sigma))+np.exp(-(x+a)**2/(2*sigma)))(x_au)
g /= np.sqrt(simps(np.abs(g)**2, x_au))
estados = np.array([g for _ in range(n)],dtype=np.complex_)
valores = np.zeros(n)
contadores = np.zeros(n)

valores_analiticos_ev = [-12.258438, -6.045418, -5.286089, -0.646627, 0.691204, 4.053229, 7.368937, 11.235521, 15.431918]
valores_analiticos_ev = np.array(valores_analiticos_ev) + adw_k0

texto_x_l = -10/2
texto_x_r = 0.7 * 10/2

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
        estados[s] /= np.sqrt(simps(np.abs(estados[s])**2, x_au))
        
        if contadores[s] % 1000 == 0:
            # calcula autoestados
            # derivada segunda
            derivada2 = (estados[s][:-2] - 2 * estados[s][1:-1] + estados[s][2:]) / dx_au**2
            psi = estados[s][1:-1]
            psi_conj = np.conjugate(psi)
            # <Psi|H|Psi>
            p_h_p = simps(psi_conj * (-0.5 * derivada2 / me_eff + v_au[1:-1] * psi), x_au[1:-1])
            # divide por <Psi|Psi> 
            #p_h_p /= A
            print(p_h_p)
            valores[s] = p_h_p.real
            
            # especificos do grafico
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_ylim([-20 + adw_k0,20 + adw_k0])
            ax.set_xlim([-6, 6])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            #plt.title("Autoestados/Autovalores Poço Suplo Assimétrico (%s)" % (propagador_titulo), fontsize=18)
            plt.xlabel("x (u. a.)", fontsize=16)
            plt.ylabel(r'$E \, (u. a.)$', fontsize=16)
            
            psif = [estados[m] for m in range(s+1)]
            psif = [2 * p / np.ptp(p) + valores[m] for m, p in enumerate(psif)]
            lines = []
            for i, p in enumerate(psif):
                line, = plt.plot(x_au, p, lw=1.0, color=tableau20[i], label=r'$|\Psi_{%d} (x,t)|^2$' % i)
                lines.append(line)
                
                if i < len(psif) - 1 and valores[i+1]-valores[i] < 2.0:
                    ax.text(texto_x_l, valores[i] - 1, r"$E_{%d} = %.5f$ eV" % (i, valores[i]))
                    ax.text(texto_x_r, valores[i] - 1, r"$%d k$ iterações" % int(contadores[i]/1000))
                else:
                    ax.text(texto_x_l, valores[i] + 0.3, r"$E_{%d} = %.5f$ eV" % (i, valores[i]))
                    ax.text(texto_x_r, valores[i] + 0.3, r"$%d k$ iterações" % int(contadores[i]/1000))
                
            linev, = plt.plot(x_au, v_au, lw=1.0, color=tableau20[n], label='$V(x)$')
            lines.append(linev)
            plt.legend(handles=lines, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=4)
            plt.show()
            
            #print("%d >>> %.4e / %.4e" % (contadores[s], valores[s], valores_analiticos_ev[s]))
            print("%d >>> %.8e" % (contadores[s], valores[s]))
            #if np.abs(valores[s] - valores_analiticos_ev[s]) < 0.000001:
            if np.abs(1-valores[s]/v_ant) < 0.0000001:
                break
            else:
                v_ant = valores[s]