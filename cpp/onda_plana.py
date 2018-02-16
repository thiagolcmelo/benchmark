# libraries
import numpy as np
from scipy.integrate import simps
import scipy.constants as cte
from scipy.sparse import diags
from scipy.linalg import inv
from scipy.fftpack import fft, ifft, fftfreq

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
delta_x = 1.0 # angstron
x0 = -20.0 # angstron

# otimizando
L = 100 # angstron
N = 256
dt = 1e-19 # s

# unidades atomicas
L_au = L / au2ang
dt_au = dt / au_t
E0_au = E0 / au2ev
delta_x_au = delta_x / au2ang
x0_au = x0 / au2ang
k0_au = np.sqrt(2 * E0_au)

# malhas direta e reciproca
dx = L / N
x_au = np.linspace(-L_au/2, L_au/2, N)
dx_au = x_au[1] - x_au[0]
k_au = fftfreq(N, d=dx_au)

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

# crank-nicolson
alpha = - dt_au * (1j / (2 * dx_au ** 2))/2.0
beta = 1.0 - dt_au * (- 1j / (dx_au ** 2))/2.0
gamma = 1.0 + dt_au * (- 1j / (dx_au ** 2))/2.0

diagonal_1 = [beta] * N
diagonal_2 = [alpha] * (N - 1)
diagonais = [diagonal_1, diagonal_2, diagonal_2]
invB = inv(diags(diagonais, [0, -1, 1]).toarray())

diagonal_3 = [gamma] * N
diagonal_4 = [-alpha] * (N - 1)
diagonais_2 = [diagonal_3, diagonal_4, diagonal_4]
C = diags(diagonais_2, [0, -1, 1]).toarray()

D = invB.dot(C)

propagador = lambda p: D.dot(psi)
propagador_titulo = "Crank-Nicolson"

# pacote de onda inicial
PN = 1/(2*np.pi*delta_x_au**2)**(1/4)
psi = PN*np.exp(1j*k0_au*x_au-(x_au-x0_au)**2/(4*delta_x_au**2))

A0 = simps(np.abs(psi)**2,x_au)

xm = x0_au
contador = 0
checar_a_cada = int((5e-18/dt) * 10)
texto_x = -L/2

devpad = 0.0
skewness = 0.0

while xm < -x0_au:
    psi = propagador(psi)
    contador += 1
    # indicadores principaus
    A = simps(np.abs(psi)**2,x_au).real
    var_norma = 100 * A / A0
    psis = np.conjugate(psi)
    xm = (simps(psis * x_au * psi,x_au)).real / A
    if xm >= -x0_au:
        # indicadores secundarios
        xm2 = simps(psis * x_au**2 * psi,x_au).real / A
        xm3 = simps(psis * x_au**3 * psi,x_au).real / A
        desvpad = np.sqrt(np.abs(xm2 - xm**2))
        skewness = (xm3 - 3*xm*desvpad**2-xm**3)/desvpad**3

print("{L:.0f} & {N} & {dt:.1e} & {dx:.3e} & {var_a:.4f} & {xmed:.4f} & {desvpad:.4f} & {skewness:.4f} & {c} & {t:.3e}".format(N=N,L=L,dt=dt,dx=L/N,t=contador*dt,var_a=var_norma,xmed=xm*au2ang,desvpad=desvpad*au2ang,skewness=skewness,c=contador))
