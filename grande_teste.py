# bibliotecas
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
import scipy.constants as cte
from scipy.fftpack import fft, ifft, fftfreq

def encontra_indicadores(xm_final):
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

    xm = 10.0
    dt_passo_aux = 0.05e-14
    dt_passo = 1e-17
    tt = 1e-18
    
    var_norma = 0.0
    desvpad = 1.0
    skewness = 0.0
    t_real = tt
    
    while np.abs(xm_final - xm * au2ang) >= 0.00005:
        # transforma para unidades atomicas
        L_au = L / au2ang
        E0_au = E0 / au2ev
        delta_x_au = delta_x / au2ang
        t_au = (tt+dt_passo) / au_t
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

        if np.abs(xm_final - xm * au2ang) >= 0.00005:
            dt_passo *= xm_final/(xm * au2ang)
            
        print(dt_passo)
    return (var_norma, xm * au2ang, desvpad * au2ang, skewness, tt+dt_passo)

print(encontra_indicadores(0.5))
