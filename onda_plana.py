#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Este módulo permite comparar os métodos de evolução temporal 
Pseudo-Espectral, Runge-Kuta e Crank-Nicolson quando aplicados a 
evolução de um pacote de onda plana livre se movendo de uma posição 
inicial até outra final
"""

# bibliotecas
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
import scipy.constants as cte
from scipy.sparse import diags
from scipy.linalg import inv
from scipy.fftpack import fft, ifft, fftfreq
import time

# grandezas de interesse em unidades atomicas
au_l = cte.value('atomic unit of length')
au_t = cte.value('atomic unit of time')
au_e = cte.value('atomic unit of energy')

# outras relacoes de interesse
ev = cte.value('electron volt')
au2ang = au_l / 1e-10
au2ev = au_e / ev

def pseudo_analitica(zi=-20.0, zf=20, E=150.0, deltaz=5.0, \
    L=250.0, N=8192):
    """
    Evolui um pacote de onda de energia `E` e dispersão `deltax`
    da posição inicial `zi` atá a posição final `zf`. 
    A evolução ocorre em um espaço de tamanho `L` partido em `N` pontos.
    Esta é uma solução pseudo analítica que assume que o problema
    analítico possui uma solução que depende apenas do resultado de uma
    quadratura, que pode ser feita numericamente com grande precisão

    Parâmetros
    ----------
    zi : float
        posição inicial em Angstrom
    zf : float 
        posição final em Angstrom
    E : float
        energia em eV
    deltaz : float
        dispersão inicial em Angstrom
    L : float
        o tamanho do espaço em Angstrom, a menos que |zf| > L/4 é melhor
        deixar o valor padrão
    N : integer
        o número de pontos no espaço, deve ser uma potência de 2
    
    Retorno
    -------
    resumo : tuple
        Os parâmetros retornados em uma tuple são:
        - `grid_z` o grid espacial em Angstrom
        - `onda_inicial` uma list com a onda inicial
        - `onda_final` uma list com a onda final
        - `norma_inicial` norma do pacote inicial
        - `norma_final` norma do pacote final
        - `desvpad` o desvio padrão do pacote final
        - `obliquidade` a obliquidade do pacote final
        - `tempo` o tempo real
        - `zf_real` a posição média final, que pode diferir de `zf`

    Exemplo
    -------
    >>> from onda_plana import *
    >>> pseudo_analitica(zi=-20.0, zf=20, L=250.0, N=512, dt=1e-18)
    """
    assert zf > zi # a onda ira da esquerda para a direita
    assert E > 0 # energia zero a onda fica parada
    assert L > 0 # espaco deve ter um tamanho nao nulo
    assert int(np.log2(N)) == np.log2(N) # deve ser potencia de 2
    assert -L/4 < zf < L/4 # posição final pode causar problemas

    # transforma para unidades atomicas
    L_au = L / au2ang
    dt_au = dt / au_t
    E_au = E / au2ev
    deltaz_au = deltaz / au2ang
    zi_au = zi / au2ang
    zf_au = zf / au2ang
    k0_au = np.sqrt(2 * E_au)

    # malhas direta e reciproca
    z_au = np.linspace(-L_au/2.0, L_au/2.0, N)
    dz_au = np.abs(z_au[1] - z_au[0])
    k_au = fftfreq(N, d=dz_au)

    # tempos
    tempo_aux = 1e-18
    tempo = 5e-18
    
    # valores iniciais
    zm_au = zi_au
    zm_au_aux = zi_au
    
    # pacote de onda inicial
    PN = 1 / (2 * np.pi * deltaz_au ** 2) ** (1 / 4)
    psi = PN * np.exp(1j*k0_au*z_au-(z_au-zi_au)**2/(4*deltaz_au**2))
    psi_inicial = np.copy(psi) # salva uma copia

    # norma inicial
    A = A0 = np.sqrt(simps(np.abs(psi) ** 2, z_au))

    # valores iniciais
    zm_au = zi_au
    desvpad_au = deltaz_au
    obliquidade = 0.0
    iteracoes = 0
    
    while np.abs(zm_au - zf_au) >= 0.00001:
        # novo tempo
        t_au = (tempo) / au_t
        
        # pacote de onda inicial
        psi = np.copy(psi_inicial)

        # IMPLEMENTACAO DE FATO DA SOLUCAO PSEUDO ANALITICA
        psi_k = fft(psi) # FFT do pacote inicial
        omega_k = k_au**2 / 2
        
        # transformada inversa do pacote de onda multiplicado
        # por uma funcao com a dependencia temporal
        psi = ifft(psi_k * np.exp(-1j * omega_k * t_au))

        # indicadores principais
        A2 = simps(np.abs(psi)**2, z_au).real # norma
        A = np.sqrt(A2)
        
        psic = np.conjugate(psi) # complexo conjugado
        zm_au = (simps(psic * x_au * psi, z_au)).real / A2
        
        # ajusta o tempo de evolucao
        if np.abs(zm_au - zf_au) >= 0.00001:
            if zm_au_aux < zf_au < zm_au or zm_au < zf_au < zm_au_aux:
                aux = (tempo_aux-tempo) / 2
            elif zf_au < zm_au and zf_au < zm_au_aux:
                aux = - abs(tempo_aux-tempo)
            elif zf_au > zm_au and zf_au > zm_au_aux:
                aux = abs(tempo_aux-tempo)
                
            tempo_aux = tempo
            tempo += aux
            zm_au_aux = zm_au
            
            continue
        
        # indicadores secundarios
        zm2 = simps(psic * z_au ** 2 * psi, z_au).real / A2
        zm3 = simps(psic * z_au ** 3 * psi, z_au).real / A2
        desvpad_au = np.sqrt(np.abs(zm2-zm_au**2))
        obliquidade = (zm3-3*zm_au*desvpad_au**2-zm_au**3)/desvpad_au**3
    
    return (z_au * au2ang, psi_inicial, psi, \
            A0, A, desvpad_au * au2ang, obliquidade, \
            tempo, zm_au * au2ang)

def evolui(zi=-20.0, zf=20, E=150.0, deltaz=5.0, metodo='pe', \
    L=100.0, N=256, dt=1e-20):
    """
    Evolui um pacote de onda de energia `E` e dispersão `deltax`
    da posição inicial `zi` atá a posição final `zf` utilizando o
    `metodo` informado. A evolução ocorre em um espaço de tamanho `L`
    partido em `N` pontos. Cada incremento no tempo é de tamanho `dt`

    Parâmetros
    ----------
    zi : float
        posição inicial em Angstrom
    zf : float 
        posição final em Angstrom
    E : float
        energia em eV
    deltaz : float
        dispersão inicial em Angstrom
    metodo : string
        o método a ser utilizado:
        - 'pe' : Pseudo-Espectral
        - 'cn' : Crank-Nicolson
        - 'rk' : Runge-Kutta
    L : float
        o tamanho do espaço em Angstrom
    N : integer
        o número de pontos no espaço, deve ser uma potência de 2
    dt : float
        o passo de tempo em segundos

    Retorno
    -------
    resumo : tuple
        Os parâmetros retornados em uma tuple são:
        - `grid_z` o grid espacial em Angstrom
        - `onda_inicial` uma list com a onda inicial
        - `onda_final` uma list com a onda final
        - `norma_inicial` norma do pacote inicial
        - `norma_final` norma do pacote final
        - `desvpad` o desvio padrão do pacote final
        - `obliquidade` a obliquidade do pacote final
        - `tempo_total` o tempo de execução
        - `iteracoes` o numero de iterações
        - `zf_real` a posição média final, que pode diferir de `zf`

    Exemplo
    -------
    >>> from onda_plana import *
    >>> evolui(zi=-20.0, zf=20, metodo='rk', L=250.0, N=512, dt=1e-18)
    """
    assert  metodo in ['pe', 'cn', 'rk'] # metodos validos
    assert zf > zi # a onda ira da esquerda para a direita
    assert E > 0 # energia zero a onda fica parada
    assert L > 0 # espaco deve ter um tamanho nao nulo
    assert int(np.log2(N)) == np.log2(N) # deve ser potencia de 2

    # transforma para unidades atomicas
    L_au = L / au2ang
    dt_au = dt / au_t
    E_au = E / au2ev
    deltaz_au = deltaz / au2ang
    zi_au = zi / au2ang
    zf_au = zf / au2ang
    k0_au = np.sqrt(2 * E_au)

    # malhas direta e reciproca
    z_au = np.linspace(-L_au/2.0, L_au/2.0, N)
    dz_au = np.abs(z_au[1] - z_au[0])
    k_au = fftfreq(N, d=dz_au)

    # comeca a contar o tempo aqui, porque algumas matrizes a seguir
    # tomam um longo tempo para inicializar e isso deve penalizar o
    # metodo envolvido
    tempo_inicial = time.time()

    # o propagador leva a funcao Psi(x,t) em Psi(x,t+dt)
    propagador = lambda p: p

    # runge-kutta ordem 4
    if metodo == 'rk':
        # parametros instaveis
        if dt_au / dz_au**2 > 0.5:
            raise Exception("Parâmetros instáveis")
        
        alpha = 1j / (2 * dz_au ** 2)
        beta = - 1j / (dz_au ** 2)
        diagonal_1 = [beta] * N
        diagonal_2 = [alpha] * (N - 1)
        diagonais = [diagonal_1, diagonal_2, diagonal_2]
        D = diags(diagonais, [0, -1, 1]).toarray()
        
        def propagador(p):
            k1 = D.dot(p)
            k2 = D.dot(p + dt_au * k1 / 2)
            k3 = D.dot(p + dt_au * k2 / 2)
            k4 = D.dot(p + dt_au * k3)
            return p + dt_au * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    # crank-nicolson
    if metodo == 'cn':
        # parametros instaveis
        if dt_au / dz_au**2 > 0.5:
            continue
        alpha = - dt_au * (1j / (2 * dz_au ** 2))/2.0
        beta = 1.0 - dt_au * (- 1j / (dz_au ** 2))/2.0
        gamma = 1.0 + dt_au * (- 1j / (dz_au ** 2))/2.0

        diagonal_1 = [beta] * N
        diagonal_2 = [alpha] * (N - 1)
        diagonais = [diagonal_1, diagonal_2, diagonal_2]
        invB = inv(diags(diagonais, [0, -1, 1]).toarray())

        diagonal_3 = [gamma] * N
        diagonal_4 = [-alpha] * (N - 1)
        diagonais_2 = [diagonal_3, diagonal_4, diagonal_4]
        C = diags(diagonais_2, [0, -1, 1]).toarray()
        
        D = invB.dot(C)
        propagador = lambda p: D.dot(p)
    
    # split step
    if metodo == 'pe':
        exp_v2 = np.ones(N, dtype=np.complex_)
        exp_t = np.exp(- 0.5j * (2 * np.pi * k_au) ** 2 * dt_au)
        propagador = lambda p: exp_v2 * ifft(exp_t * fft(exp_v2 * p))
        
    # pacote de onda inicial
    PN = 1 / (2 * np.pi * deltaz_au ** 2) ** (1 / 4)
    psi = PN * np.exp(1j*k0_au*z_au-(z_au-zi_au)**2/(4*deltaz_au**2))
    psi_inicial = np.copy(psi) # salva uma copia

    # norma inicial
    A = A0 = np.sqrt(simps(np.abs(psi) ** 2, z_au))

    # valores iniciais
    zm_au = zi_au
    desvpad_au = deltaz_au
    obliquidade = 0.0
    iteracoes = 0

    while zm_au < zf_au:
        psi = propagador(psi)
        iteracoes += 1

        # se a onda andar para o lado errado, inverte o vetor de onda
        # espera-se que isso ocorra no maximo uma vez (no inicio)
        if zm_au < zi_au:
            k0_au *= -1.0
            psi = PN*np.exp(1j*k0_au*z_au-(z_au-zi_au)**2/(4*deltaz**2))
            zm_au = zi_au
            continue

        # indicadores principais
        A2 = simps(np.abs(psi) ** 2, z_au).real
        A = np.sqrt(A2)
        
        # complexo conjugado de psi
        psic = np.conjugate(psi)

        # posicao media de psi
        zm_au = (simps(psic * z_au * psi, z_au)).real / A2

        # termina de contar aqui
        tempo_final = time.time()
        
        # se atingir a posicao inicial ou se o tempo estourar
        if zm_au >= zf_au or tempo_final - tempo_inicial > 1000:
            # indicadores secundarios
            zm2 = simps(psic * z_au ** 2 * psi, z_au).real / A2
            zm3 = simps(psic * z_au ** 3 * psi, z_au).real / A2
            desvpad_au = np.sqrt(np.abs(zm2-zm_au**2))
            obliquidade = (zm3-3*zm_au*desvpad_au**2-zm_au**3)\
                /desvpad_au**3
            
            # se o tempo estourou
            if tempo_final - tempo_inicial > 1000:
                break

    tempo_total_programa = tempo_final - tempo_inicial
    
    return (z_au * au2ang, psi_inicial, psi, \
            A0, A, desvpad_au * au2ang, obliquidade,
            tempo_total_programa, iteracoes, zm_au * au2ang)

# constantes do problema
E0 = 150.0 # eV
delta_x = 5.0 # angstron
x0 = -20.0 # angstron

def solucao_pseudo_analitica(xm_final):
    """
    esta funcao ira buscar um tempo que a onda
    levaria para atingir a posicao xm_final
    e retorna os parametros
    - conservacao da norma
    - xm_medio encontrado (que difere um pouco de xm_final)
    - desvio padrao da onda
    - skewness da onda
    - tempo que a onda leva para atingir xm_final
    
    xm_final : float
        um float indicando a posicao final
        media esperada em Angstrom
    """
    L = 250 # angstron
    N = 8192 # pontos
    
    if xm_final < -L/4 or xm_final > L/4:
        return (0, 0, 0, 0, 0)

    # valores iniciais
    xm = 10.0
    xm_a_aux = x0
    dt_passo_aux = 1e-18
    dt_passo = 5e-18
    tt = 0.8e-14
    var_norma = 0.0
    desvpad = 1.0
    skewness = 0.0
    t_real = tt
    
    while np.abs(xm_final - xm * au2ang) >= 0.00001:
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
        xm_a = xm * au2ang
        
        # ajusta o tempo de evolucao
        if np.abs(xm_final - xm_a) >= 0.00001:
            if xm_a_aux < xm_final < xm_a or xm_a < xm_final < xm_a_aux:
                aux = (dt_passo_aux-dt_passo) / 2
            elif xm_final < xm_a and xm_final < xm_a_aux:
                aux = - abs(dt_passo_aux-dt_passo)
            elif xm_final > xm_a and xm_final > xm_a_aux:
                aux = abs(dt_passo_aux-dt_passo)
                
            dt_passo_aux = dt_passo
            dt_passo += aux
            xm_a_aux = xm_a
            
            continue
        
        # indicadores secundarios
        xm2 = simps(psis * x_au**2 * psi,x_au).real / A
        xm3 = simps(psis * x_au**3 * psi,x_au).real / A
        desvpad = np.sqrt(np.abs(xm2 - xm**2)) # desvio padrao
        skewness = (xm3 - 3*xm*desvpad**2-xm**3)/desvpad**3 # obliquidade
            
    return (var_norma, xm * au2ang, desvpad * au2ang, skewness, tt+dt_passo)

for metodo in ['pe', 'cn', 'rk']:
    for L in np.linspace(100,1000,7):
        for N in [2**n for n in range(8,13)]:
            for dt in [1e-20, 5e-20, 1e-19, 5e-19, 1e-18, \
                       5e-18, 1e-17, 5e-17, 1e-16, 5e-16]:
                # dt = 1e-20 e 5e-20 usado apenas para o
                # metodo pseudo-espectral
                if dt < 1e-19 and metodo != 'pe':
                    continue
                    
                # unidades atomicas
                L_au = L / au2ang
                dt_au = dt / au_t
                E0_au = E0 / au2ev
                delta_x_au = delta_x / au2ang
                x0_au = x0 / au2ang
                k0_au = np.sqrt(2 * E0_au)

                # malhas direta e reciproca
                dx = L   / (N-1)
                x_au = np.linspace(-L_au/2.0, L_au/2.0, N)
                dx_au = np.abs(x_au[1] - x_au[0])
                k_au = fftfreq(N, d=dx_au)

                # comeca a contar o tempo aqui
                # porque inicializar algumas matrizes
                # acaba levando muito tempo
                inicio = time.time()

                # runge-kutta ordem 4
                if metodo == 'rk':
                    # parametros instaveis
                    if dt_au / dx_au**2 > 0.5:
                        continue
                    alpha = 1j / (2 * dx_au ** 2)
                    beta = - 1j / (dx_au ** 2)
                    diagonal_1 = [beta] * N
                    diagonal_2 = [alpha] * (N - 1)
                    diagonais = [diagonal_1, diagonal_2, diagonal_2]
                    D = diags(diagonais, [0, -1, 1]).toarray()
                    def propagador(p):
                        k1 = D.dot(p)
                        k2 = D.dot(p + dt_au * k1 / 2)
                        k3 = D.dot(p + dt_au * k2 / 2)
                        k4 = D.dot(p + dt_au * k3)
                        return p + dt_au * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                
                # crank-nicolson
                if metodo == 'cn':
                    # parametros instaveis
                    if dt_au / dx_au**2 > 0.5:
                        continue
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
                    propagador = lambda p: D.dot(p)
                
                # split step
                if metodo == 'pe':
                    exp_v2 = np.ones(N, dtype=np.complex_)
                    exp_t = np.exp(- 0.5j * (2 * np.pi * k_au) ** 2 * dt_au)
                    propagador = lambda p: exp_v2 * ifft(exp_t * fft(exp_v2 * p))
                    
                # pacote de onda inicial
                PN = 1/(2*np.pi*delta_x_au**2)**(1/4)
                psi = PN*np.exp(1j*k0_au*x_au-(x_au-x0_au)**2/(4*delta_x_au**2))
                A0 = simps(np.abs(psi)**2,x_au)

                # valores iniciais
                xm = x0_au
                desvpad = delta_x_au
                skewness = 0.0
                contador = 0

                while xm < -x0_au:
                    psi = propagador(psi)
                    contador += 1

                    if xm < x0_au:
                        k0_au *= -1.0
                        psi = PN*np.exp(1j*k0_au*x_au-(x_au-x0_au)**2/(4*delta_x_au**2))
                        xm = x0_au
                        continue

                    # indicadores principais
                    A = simps(np.abs(psi)**2,x_au).real
                    var_norma = 100 * A / A0
                    psis = np.conjugate(psi)
                    xm = (simps(psis * x_au * psi,x_au)).real / A
                    final = time.time()
                    
                    # se atingir 1000 segundos para a evolucao
                    # se sim calcula os indicadores
                    if final - inicio > 1000:
                        # indicadores secundarios
                        xm2 = simps(psis * x_au**2 * psi,x_au).real / A
                        xm3 = simps(psis * x_au**3 * psi,x_au).real / A
                        desvpad = np.sqrt(np.abs(xm2 - xm**2))
                        skewness = (xm3 - 3*xm*desvpad**2-xm**3)/desvpad**3
                        break

                    # checa se a posicao final foi aingida
                    # se sim calcula os indicadores
                    if xm >= -x0_au:
                        # indicadores secundarios
                        xm2 = simps(psis * x_au**2 * psi,x_au).real / A
                        xm3 = simps(psis * x_au**3 * psi,x_au).real / A
                        desvpad = np.sqrt(np.abs(xm2 - xm**2))
                        skewness = (xm3 - 3*xm*desvpad**2-xm**3)/desvpad**3

                tempo_total_programa = final - inicio
                var_norma_ana, xm_ana, desvpad_ana, skewness_ana, tt_ana = solucao_pseudo_analitica(xm*au2ang)
                params = "{0:.0f},{1},{2:.1e},{3:.3e}".format(L,N,dt,L/N)
                qualid = "{0:.8f},{1:.8f},{2:.8f}".format(var_norma, desvpad*au2ang, skewness)
                print("{L:.0f},{N},{dt:.1e},{dx:.3e},{var_a:.8f},{xmed:.8f},{desvpad:.8f},{skewness:.8f},{c},{t:.3e},{time:.10f},{var_a_ana:.8f},{xmed_ana:.8f},{desvpad_ana:.8f},{skewness_ana:.8f},{tt_ana:.10e}".format(N=N,L=L,dt=dt,dx=L/N,t=contador*dt,var_a=var_norma,xmed=xm*au2ang,desvpad=desvpad*au2ang,skewness=skewness,c=contador,time=tempo_total_programa,var_a_ana=var_norma_ana,xmed_ana=xm_ana,desvpad_ana=desvpad_ana,skewness_ana=skewness_ana,tt_ana=tt_ana))