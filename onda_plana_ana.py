# bibliotecas
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
import scipy.constants as cte
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
delta_x = 5.0 # angstron
x0 = -20.0 # angstron

# otimizando
L = 150 # angstron
N = 8192 # pontos
tt = 2.17395e-14 # tempo total de evolução em segundos

# transforma para unidades atomicas
L_au = L / au2ang
E0_au = E0 / au2ev
delta_x_au = delta_x / au2ang
t_au = tt / au_t
x0_au = x0 / au2ang
k0_au = np.sqrt(2 * E0_au)

# malhas direta e reciproca
dx = L / (N-1)
x_au = np.linspace(-L_au/2.0, L_au/2.0, N)
dx_au = np.abs(x_au[1] - x_au[0])
k_au = fftfreq(N, d=dx_au)

# pacote de onda inicial
PN = 1/(2*np.pi*delta_x_au**2)**(1/4)
psi = PN*np.exp(1j*k0_au*x_au-(x_au-x0_au)**2/(4*delta_x_au**2))
A0 = simps(np.abs(psi)**2,x_au) # norma inicial
psi0 = np.copy(psi) # salva uma copia da onda inicial

# valores iniciais
xm = x0_au
contador = 0
texto_x = -L/2

# IMPLEMENTACAO DE FATO DA SOLUCAO PSEUDO ANALITICA
psi_k = fft(psi) # FFT do pacote inicial
omega_k = k_au**2 / 2
# transformada inversa do pacote de onda multiplicado
# por uma funcao com a dependencia temporal
psi = ifft(psi_k * np.exp(-1j * omega_k * t_au))

# indicadores principaus
A = simps(np.abs(psi)**2,x_au).real # norma
var_norma = 100 * A / A0 # conservacao da norma
psis = np.conjugate(psi) # complexo conjugado
xm = (simps(psis * x_au * psi,x_au)).real / A # posicao media final <x>

# indicadores secundarios
xm2 = simps(psis * x_au**2 * psi,x_au).real / A
xm3 = simps(psis * x_au**3 * psi,x_au).real / A
desvpad = np.sqrt(np.abs(xm2 - xm**2)) # desvio padrao
skewness = (xm3 - 3*xm*desvpad**2-xm**3)/desvpad**3 # obliquidade

# especificos do grafico
def as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))
    
plt.rcParams['savefig.dpi'] = 75
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 14, 8
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "computer modern sans serif"
plt.rcParams['text.usetex']=True
plt.rcParams['text.latex.unicode']=True

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_ylim([-0.005,0.25])
ax.text(-14, 0.22, r"$\mathbf{L = %.2f}$ \textbf{\AA}" % (L), color='red', fontsize=24)
ax.text(texto_x, 0.235, r"N = %d pontos" % N)
ax.text(texto_x, 0.21, r"$\Delta x = %.5f$ \AA" % (dx))
ax.text(texto_x, 0.18, r"*$(t_f-t_i) = %s$ s" % (as_si(t_au * au_t, 2)))
ax.text(texto_x, 0.15, r"$\langle x_i \rangle = %.5f$ \AA" % (x0_au * au2ang))
ax.text(texto_x, 0.12, r"$\langle x_f \rangle = %.5f$ \AA" % (xm * au2ang))
ax.text(texto_x, 0.09, r"$\sigma_i = %.5f$ \AA" % (delta_x_au * au2ang))
ax.text(texto_x, 0.06, r"$\sigma_f = %.5f$ \AA" % (desvpad * au2ang))
ax.text(texto_x, 0.03, r"$\gamma_i = \gamma_f = %.5f$" % np.abs(skewness))
ax.text(texto_x, 0.005, r"$A_f/A_i = %.2f$ \%%" % (var_norma))
plt.xlabel("x (\AA)", fontsize=24)
plt.ylabel(r'$|\Psi (x,t)|^2$', fontsize=24)
line1, = plt.plot(x_au * au2ang, np.abs(psi), lw=2.0, \
    color=(31/255, 119/255, 180/255), label='$\Psi(x,t_f)$')
line2, = plt.plot(x_au * au2ang, np.abs(psi0), lw=2.0, \
    color=(174/255, 199/255, 232/255), label='$\Psi(x,t_i)$')
plt.legend(handles=[line1,line2], loc=1)
plt.show()