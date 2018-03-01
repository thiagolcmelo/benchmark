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
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import os, time
from multiprocessing import Pool, TimeoutError
import logging

logger = logging.getLogger('onda_plana_logger')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(\
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# configuracao do matplotlib
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

# grandezas de interesse em unidades atomicas
au_l = cte.value('atomic unit of length')
au_t = cte.value('atomic unit of time')
au_e = cte.value('atomic unit of energy')

# outras relacoes de interesse
ev = cte.value('electron volt')
au2ang = au_l / 1e-10
au2ev = au_e / ev

def evolucao_analitica(zi=-20.0, zf=20, E=150.0, deltaz=5.0, \
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
    resumo : dict
        Os parâmetros retornados em um dict são:
        - `L` o tamanho do espaço
        - `N` o número de pontos
        - `dt` o passo de tempo
        - `metodo` o método utilizado
        - `grid_z` o grid espacial em Angstrom
        - `onda_inicial` uma list com a onda inicial
        - `onda_final` uma list com a onda final
        - `norma_inicial` norma do pacote inicial
        - `norma_final` norma do pacote final
        - `conservacao` 100 * norma_final / norma_inicial
        - `desvpad` o desvio padrão do pacote final
        - `obliquidade` a obliquidade do pacote final
        - `tempo` o tempo necessario para ir de zi ate zf
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
        zm_au = (simps(psic * z_au * psi, z_au)).real / A2
        
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
    
    return {
        'L': L,
        'N': N,
        'dt': dt,
        'metodo': metodo,
        'grid_z': z_au * au2ang,
        'onda_inicial': psi_inicial,
        'onda_final': psi,
        'norma_inicial': A0,
        'norma_final': A,
        'conservacao': 100 * A / A0,
        'desvpad': desvpad_au * au2ang,
        'obliquidade': obliquidade,
        'tempo': tempo,
        'zf_real': zm_au * au2ang,
    }

def evolucao_numerica(zi=-20.0, zf=20, E=150.0, deltaz=5.0, \
    metodo='pe', L=100.0, N=256, dt=1e-20):
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
    resumo : dict
        Os parâmetros retornados em um dict são:
        - `L` o tamanho do espaço
        - `N` o número de pontos
        - `dt` o passo de tempo
        - `metodo` o método utilizado
        - `grid_z` o grid espacial em Angstrom
        - `onda_inicial` uma list com a onda inicial
        - `onda_final` uma list com a onda final
        - `norma_inicial` norma do pacote inicial
        - `norma_final` norma do pacote final
        - `conservacao` 100 * norma_final / norma_inicial 
        - `desvpad` o desvio padrão do pacote final em Angstrom
        - `obliquidade` a obliquidade do pacote final
        - `tempo_total` o tempo de execução em segundos
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
            raise Exception("Parâmetros instáveis")
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
    
    return {
        'L': L,
        'N': N,
        'dt': dt,
        'metodo': metodo,
        'grid_z': z_au * au2ang,
        'onda_inicial': psi_inicial,
        'onda_final': psi,
        'norma_inicial': A0,
        'norma_final': A,
        'conservacao': 100 * A / A0,
        'desvpad': desvpad_au * au2ang,
        'obliquidade': obliquidade,
        'tempo_total': tempo_total_programa,
        'iteracoes': iteracoes,
        'zf_real': zm_au * au2ang
    }

if __name__ == '__main__':

    metodos = ['pe', 'cn', 'rk']
    passos = [1e-20, 5e-20, 1e-19, 5e-19, 1e-18, 5e-18, 1e-17, \
        5e-17, 1e-16, 5e-16]
    passos = [1e-18, 5e-18, 1e-17, \
        5e-17, 1e-16, 5e-16]
    
    resultados = []
    combinacoes = []

    for metodo in metodos:
        for L in np.linspace(100,1000,7):
            for N in [2**n for n in range(8,13)]:
                for dt in passos:
                    if dt < 1e-19 and metodo != 'pe':
                        continue
                    combinacoes.append((metodo, L, N, dt))

    def evolui_ponto(combinacao):
        try:
            metodo, L, N, dt = combinacao
            
            res = evolucao_numerica(L=L, N=N, \
                dt=dt, metodo=metodo)
            zf_real = res['zf_real']
            
            res_ana = evolucao_analitica(zf=zf_real)
            
            for k in res_ana.keys():
                res[k + '_ana'] = res_ana[k]
            
            mensagem = "%s: L=%d, N=%d, dt=%.2e, " + \
                        "A/A0=%.5f, S=%.4f, G=%.4f, " + \
                        "A/A0_ana=%.5f, S_ana=%.4f, G_ana=%.4f, " + \
                        "tempo=%.5f"
            mensagem = mensagem % (metodo, L, N, dt,\
                res['conservacao'], res['desvpad'], \
                res['obliquidade'], res['conservacao_ana'], \
                res['desvpad_ana'], res['obliquidade_ana'],
                res['tempo_total'])
            logger.info(mensagem)

            return res
        except Exception as err:
            logger.error("Falha em %s: L=%d, N=%d, dt=%.2e" % \
                (metodo, L, N, dt))
            logger.error(str(err))
            return {}

    pool = Pool(processes=8)
    resultados = pool.map(evolui_ponto, combinacoes)

    resultados = pd.DataFrame(resultados)
    
    pec = resultados.loc[resultados['metodo'] == 'pe']
    cnc = resultados.loc[resultados['metodo'] == 'cn']
    rkc = resultados.loc[resultados['metodo'] == 'rk']
    pec.to_csv('onda_plana_resultados_pec.csv')
    cnc.to_csv('onda_plana_resultados_cnc.csv')
    rkc.to_csv('onda_plana_resultados_rkc.csv')

    # ajuste de escala dos parametros de qualidade
    scaler = StandardScaler()
    cols = ['devpad', 'obliquidade', 'conservacao', \
        'devpad_ana', 'obliquidade_ana', 'conservacao_ana']
    pec[cols] = scaler.fit_transform(pec[cols])
    rkc[cols] = scaler.fit_transform(rkc[cols])
    cnc[cols] = scaler.fit_transform(cnc[cols])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
    with pd.plotting.plot_params.use('x_compat', True):
        def minkowski(line, p=3):
            x_num = [[line['desvpad'], line['obliquidade'], \
                line['conservacao']]]
            x_ana = [[line['desvpad_ana'], line['obliquidade_ana'], \
                line['conservacao_ana']]]
            dist = cdist(XA=x_num, XB=x_ana, metric='minkowski', p=p)
            return dist[0][0]

        pec['minkowski'] = pec.apply(minkowski, axis=1)
        pec.plot(x='tempo_total', y='minkowski', kind='scatter', \
            loglog=True, color='r', ax=ax1)

        rkc['minkowski'] = rkc.apply(minkowski, axis=1)
        rkc.plot(x='tempo_total', y='minkowski', kind='scatter', \
            loglog=True, color='g', ax=ax2)

        cnc['minkowski'] = cnc.apply(minkowski, axis=1)
        cnc.plot(x='tempo_total', y='minkowski', kind='scatter', \
            loglog=True, color='b', ax=ax3)

    ax1.title.set_text('Pseudo-Espectral')
    ax2.title.set_text('Runge-Kutta')
    ax3.title.set_text('Crank-Nicolson')
    ax1.set_ylabel('Minkowski (p=3)')
    ax2.set_ylabel('Minkowski (p=3)')
    ax3.set_ylabel('Minkowski (p=3)')
    ax1.set_xlabel('Tempo total (s)')
    ax2.set_xlabel('Tempo total (s)')
    ax3.set_xlabel('Tempo total (s)')
    plt.show()