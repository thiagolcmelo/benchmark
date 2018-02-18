# libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
import scipy.constants as cte
from scipy.sparse import diags
from scipy.linalg import inv
from scipy.fftpack import fft, ifft, fftfreq

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

# constantes do problema
E0 = 150.0 # eV
V0 = [100.0, 200.0] # eV
delta_x = 5.0 # angstron
x0 = -30.0 # angstron
xf = -40.0 # angstron

# otimizando
L = 250 # angstron
N = 2048
hN = int(N/2)
dt = 1e-19 # s

# unidades atomicas
L_au = L / au2ang
dt_au = dt / au_t
E0_au = E0 / au2ev
V0_au = np.array(V0) / au2ev
delta_x_au = delta_x / au2ang
x0_au = x0 / au2ang
xf_au = xf / au2ang
k0_au = np.sqrt(2 * E0_au)

# malhas direta e reciproca
dx = L / (N-1)
x_au = np.linspace(-L_au/2.0, L_au/2.0, N)
dx_au = np.abs(x_au[1] - x_au[0])
k_au = fftfreq(N, d=dx_au)

for v0_au in V0_au:
    v_au = np.vectorize(lambda x: 0.0 if x < 0.0 else v0_au)(x_au)
    
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
    
    # pacote de onda inicial
    PN = 1/(2*np.pi*delta_x_au**2)**(1/4)
    psi = PN*np.exp(1j*k0_au*x_au-(x_au-x0_au)**2/(4*delta_x_au**2))
    A0 = simps(np.abs(psi)**2,x_au)
    
    xm_r = xm = x0_au
    contador = 0
    texto_x = -L/2
    
    while xm_r > xf_au:
        #print(xm*au2ang)
        psi = propagador(psi)
        contador += 1
    
        # if xm < x0_au:
        #     k0_au *= -1.0
        #     psi = PN*np.exp(1j*k0_au*x_au-(x_au-x0_au)**2/(4*delta_x_au**2))
        #     xm = x0_au
        #     continue
    
        # indicadores principaus
        A = simps(np.abs(psi)**2,x_au).real
        rA = simps(np.abs(psi[:hN])**2,x_au[:hN]).real
        
        var_norma = 100 * A / A0
        psis = np.conjugate(psi)
        # xm = (simps(psis * x_au * psi,x_au)).real / A
        
        xm_r = (simps(psis[:hN] * x_au[:hN] * psi[:hN], x_au[:hN])).real / rA
        
        if contador % 100 == 0:
            tA = simps(np.abs(psi[hN:])**2,x_au[hN:]).real
            xm_t = (simps(psis[hN:] * x_au[hN:] * psi[hN:], x_au[hN:])).real / tA    
            R = (simps(np.abs(psi[:hN])**2, x_au[:hN])).real / A
            T = (simps(np.abs(psi[hN:])**2, x_au[hN:])).real / A
            
            #xm2 = simps(psis * x_au**2 * psi,x_au).real / A
            #xm3 = simps(psis * x_au**3 * psi,x_au).real / A
            
            #desvpad = np.sqrt(np.abs(xm2 - xm**2))
            #skewness = (xm3 - 3*xm*desvpad**2-xm**3)/desvpad**3
    
            xm2_r = simps(psis[:hN] * x_au[:hN]**2 * psi[:hN], x_au[:hN]).real / rA
            xm2_t = simps(psis[hN:] * x_au[hN:]**2 * psi[hN:], x_au[hN:]).real / tA
            desvpad_r = np.sqrt(np.abs(xm2_r - xm_r**2))
            desvpad_t = np.sqrt(np.abs(xm2_t - xm_t**2))
    
            # especificos do grafico
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_ylim([-10,210])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            # ax.text(texto_x, 130, r"$A/A_0 = %.3f$ \%%" % (var_norma))
            # ax.text(texto_x, 110, r"$R = %.3f$ \%%" % (100 * R))
            # ax.text(texto_x, 90, r"$T = %.3f$ \%%" % (100 * T))
            # ax.text(texto_x, 70, r"$\langle x_R \rangle = %.2f$ \AA" % (xm_r * au2ang))
            # ax.text(texto_x, 50, r"$\sigma_R = %.2f$ \AA" % (desvpad_r * au2ang))
            # ax.text(texto_x, 30, r"$\langle x_T \rangle = %.2f$ \AA" % (xm_t * au2ang))
            # ax.text(texto_x, 10, r"$\sigma_T = %.2f$ \AA" % (desvpad_t * au2ang))
            
            plt.title("Onda incidindo em barreira (%s)" % (propagador_titulo), fontsize=18)
            plt.xlabel("x (\AA)", fontsize=16)
            plt.ylabel(r'$E \, (eV)$', fontsize=16)
            
            psif = np.abs(psi)
            psif = 25.0 * psif / np.max(psif) + E0
            
            line1, = plt.plot(x_au * au2ang, psif, lw=1.0, color=tableau20[0], label=r'$|\Psi (x,t)|^2$')
            line2, = plt.plot(x_au * au2ang, v_au * au2ev, lw=1.0, color=tableau20[1], label='$V_0(x)$')
            
            # ni = hN - int(N/6)
            # nf = hN + int(N/6)
            # line1, = plt.plot(x_au[ni:nf] * au2ang, psif[ni:nf], lw=1.0, color=tableau20[0], label=r'$|\Psi (x,t)|^2$')
            # line2, = plt.plot(x_au[ni:nf] * au2ang, v_au[ni:nf] * au2ev, lw=1.0, color=tableau20[1], label='$V_0(x)$')
            
            plt.legend(handles=[line1, line2], loc=1)
            plt.legend()
            plt.show()
    
            #print("{L:.0f} & {N} & {dt:.1e} & {dx:.3e} & {var_a:.4f} & {xmed:.4f} & {desvpad:.4f} & {skewness:.4f} & {c} & {t:.3e}".format(N=N,L=L,dt=dt,dx=L/N,t=contador*dt,var_a=var_norma,xmed=xm*au2ang,desvpad=desvpad*au2ang,skewness=skewness,c=contador))
    
