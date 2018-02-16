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
delta_x = 5.0 # angstron
x0 = -20.0 # angstron

# otimizando
L = 100 # angstron
N = 512
dt = 1e-19 # s


# divisor esperto (?)
de = lambda x: 2 if int((x/(10**(int(np.log10(x))-1)))%2) == 0 else 5

combinacoes = []

for L in np.linspace(100,1000,7):
    for N in [8192]:#[2**n for n in range(8,12)]:
        for dt in [1e-20, 5e-20, 1e-19, 5e-19, 1e-18, 5e-18]:
            # unidades atomicas
            L_au = L / au2ang
            dt_au = dt / au_t
            E0_au = E0 / au2ev
            delta_x_au = delta_x / au2ang
            x0_au = x0 / au2ang
            k0_au = np.sqrt(2 * E0_au)

            # malhas direta e reciproca
            dx = L / (N-1)
            x_au = np.linspace(-L_au/2.0, L_au/2.0, N)
            dx_au = np.abs(x_au[1] - x_au[0])
            k_au = fftfreq(N, d=dx_au)

            # while dt_au / dx_au**2 > 0.5:
            #     dt /= de(dt)
            #     dt_au = dt / au_t

            # comb = "%.0f-%d-%.2e" % (L, N, dt)
            # if comb in combinacoes:
            #     continue
            # combinacoes.append(comb)

            # # runge-kutta ordem 4
            # alpha = 1j / (2 * dx_au ** 2)
            # beta = - 1j / (dx_au ** 2)
            # diagonal_1 = [beta] * N
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
            # beta = 1.0 - dt_au * (- 1j / (dx_au ** 2))/2.0
            # gamma = 1.0 + dt_au * (- 1j / (dx_au ** 2))/2.0
            # diagonal_1 = [beta] * N
            # diagonal_2 = [alpha] * (N - 1)
            # diagonais = [diagonal_1, diagonal_2, diagonal_2]
            # invB = inv(diags(diagonais, [0, -1, 1]).toarray())
            # diagonal_3 = [gamma] * N
            # diagonal_4 = [-alpha] * (N - 1)
            # diagonais_2 = [diagonal_3, diagonal_4, diagonal_4]
            # C = diags(diagonais_2, [0, -1, 1]).toarray()
            # D = invB.dot(C)
            # propagador = lambda p: D.dot(psi)
            # propagador_titulo = "Crank-Nicolson"

            # split step
            exp_v2 = np.ones(N, dtype=np.complex_)
            exp_t = np.exp(- 0.5j * (2 * np.pi * k_au) ** 2 * dt_au)
            propagador = lambda p: exp_v2 * ifft(exp_t * fft(exp_v2 * p))
            propagador_titulo = "Split-Step"


            # pacote de onda inicial
            PN = 1/(2*np.pi*delta_x_au**2)**(1/4)
            psi = PN*np.exp(1j*k0_au*x_au-(x_au-x0_au)**2/(4*delta_x_au**2))
            A0 = simps(np.abs(psi)**2,x_au)

            xm = x0_au
            contador = 0
            checar_a_cada = int((5e-18/dt) * 10)
            texto_x = -L/2

            while xm < -x0_au:
                #print(xm*au2ang)
                psi = propagador(psi)
                contador += 1

                if xm < x0_au:
                    k0_au *= -1.0
                    psi = PN*np.exp(1j*k0_au*x_au-(x_au-x0_au)**2/(4*delta_x_au**2))
                    xm = x0_au
                    continue

                # indicadores principaus
                A = simps(np.abs(psi)**2,x_au).real
                var_norma = 100 * A / A0
                psis = np.conjugate(psi)
                xm = (simps(psis * x_au * psi,x_au)).real / A

                #if contador % checar_a_cada == 0 or xm >= -x0_au:
                if xm >= -x0_au:
                    # indicadores secundarios
                    xm2 = simps(psis * x_au**2 * psi,x_au).real / A
                    xm3 = simps(psis * x_au**3 * psi,x_au).real / A
                    desvpad = np.sqrt(np.abs(xm2 - xm**2))
                    skewness = (xm3 - 3*xm*desvpad**2-xm**3)/desvpad**3

                    # # especificos do grafico
                    # fig = plt.figure()
                    # ax = fig.add_subplot(1, 1, 1)
                    # ax.set_ylim([-0.05,0.5])
                    # ax.spines["top"].set_visible(False)
                    # ax.spines["right"].set_visible(False)
                    # ax.text(texto_x, 0.50, r"$L = %.1f$ \AA" % (L))
                    # ax.text(texto_x, 0.45, r"N = %d" % N)
                    # ax.text(texto_x, 0.40, r"$\Delta t = %.2e$ s" % (dt))
                    # ax.text(texto_x, 0.35, r"$\Delta x = %.2e$ \AA" % (dx))
                    # ax.text(texto_x, 0.30, r"$A/A_0 = %.3f$ \%%" % (var_norma))
                    # ax.text(texto_x, 0.25, r"$\langle x \rangle = %.2f$ \AA" % (xm * au2ang))
                    # ax.text(texto_x, 0.20, r"$3 \sigma = %.2f$ \AA" % (3 * desvpad * au2ang))
                    # ax.text(texto_x, 0.15, r"$\gamma = %.2f$" % (skewness))
                    # ax.text(texto_x, 0.10, r"$Iter = %d$" % (contador))
                    # ax.text(texto_x, 0.05, r"$t = %.2e$ s" % (contador*dt))
                    # plt.title("Pacote de Onda Plana (%s)" % (propagador_titulo), fontsize=18)
                    # plt.xlabel("x (\AA)", fontsize=16)
                    # plt.ylabel(r'$|\Psi (x,t)|^2$', fontsize=16)
                    # line, = plt.plot(x_au * au2ang, np.abs(psi), lw=2.0, color=tableau20[0], label='$\Psi(x,%.2e)$' % (dt))
                    # plt.legend(handles=[line], loc=1)
                    # plt.legend()
                    # plt.show()

                    print("{L:.0f} & {N} & {dt:.1e} & {dx:.3e} & {var_a:.4f} & {xmed:.4f} & {desvpad:.4f} & {skewness:.4f} & {c} & {t:.3e}".format(N=N,L=L,dt=dt,dx=L/N,t=contador*dt,var_a=var_norma,xmed=xm*au2ang,desvpad=desvpad*au2ang,skewness=skewness,c=contador))

