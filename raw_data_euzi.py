
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
plt.rcParams['figure.figsize'] = 14, 8


# In[76]:


n = 6
files = np.load('res_numeric_iter.npz')
res_numeric_iter = files['arr_0']
files = np.load('res_numeric_time.npz')
res_numeric_time = files['arr_0']


# In[10]:


# for j in range(n):
#     chebyshev = []
#     for res in res_numeric_iter:
#         chebyshev.append(res['chebyshev'][j])
#     plt.semilogy([i * 1e5 for i in range(1,11)], chebyshev, label='Chebyshev_state_%d' % j)

for j in range(n):
    chebyshev = []
    for res in res_numeric_time:
        chebyshev.append(res['chebyshev'][j])
    plt.semilogy([60 * i for i in range(1,6)], chebyshev, label='sqeuclidian_state_%d' % j)
plt.legend()
plt.show()


# In[73]:


res_iterations = []
for j in range(n):
    precisions = []
    chebyshev = []
    seuclidean = []
    sqeuclidean = []
    for res in res_numeric_iter:
        precisions.append(res['precisions'][j])
        chebyshev.append(res['chebyshev'][j])
        seuclidean.append(res['seuclidean'][j])
        sqeuclidean.append(res['sqeuclidean'][j])
    iterations = [i * 1e5 for i in range(1,11)]
    for i, it in enumerate(iterations):
        res_iterations.append({
            'state': j,
            'iterations': it,
            'eigenvalue_precision': precisions[i],
            'eigenstate_chebyshev': chebyshev[i],
            'eigenstate_seuclidean': seuclidean[i],
            'eigenstate_sqeuclidean': sqeuclidean[i],
        })
res_iterations = pd.DataFrame(res_iterations)
for j in range(6):
    j_es = res_iterations.loc[res_iterations['state'] == j]
    plt.semilogy(j_es['iterations'], j_es['eigenstate_sqeuclidean'], label='state_%d' % j)
plt.legend()
plt.show()
chebyshev = res_iterations.pivot(index='iterations', columns='state', values='eigenstate_chebyshev').reset_index(level=0)
sqeuclidean = res_iterations.pivot(index='iterations', columns='state', values='eigenstate_sqeuclidean').reset_index(level=0)
chebyshev.to_csv('oscilador_harmonica_quantico_por_iteracoes_chebyshev.csv', index=False)
sqeuclidean.to_csv('oscilador_harmonica_quantico_por_iteracoes_sqeuclidean.csv', index=False)


# In[77]:


res_timers = []
for j in range(n):
    precisions = []
    chebyshev = []
    seuclidean = []
    sqeuclidean = []
    for res in res_numeric_time:
        precisions.append(res['precisions'][j])
        chebyshev.append(res['chebyshev'][j])
        seuclidean.append(res['seuclidean'][j])
        sqeuclidean.append(res['sqeuclidean'][j])
    timers = [60 * i for i in range(1,6)]
    for i, tim in enumerate(timers):
        res_timers.append({
            'state': j,
            'time': tim,
            'eigenvalue_precision': precisions[i],
            'eigenstate_chebyshev': chebyshev[i],
            'eigenstate_seuclidean': seuclidean[i],
            'eigenstate_sqeuclidean': sqeuclidean[i],
        })
res_timers = pd.DataFrame(res_timers)
for j in range(6):
    j_es = res_timers.loc[res_timers['state'] == j]
    plt.semilogy(j_es['time'], j_es['eigenstate_sqeuclidean'], label='state_%d' % j)
plt.legend()
plt.show()
chebyshev = res_timers.pivot(index='time', columns='state', values='eigenstate_chebyshev').reset_index(level=0)
sqeuclidean = res_timers.pivot(index='time', columns='state', values='eigenstate_sqeuclidean').reset_index(level=0)
chebyshev.to_csv('oscilador_harmonica_quantico_por_tempo_chebyshev.csv', index=False)
sqeuclidean.to_csv('oscilador_harmonica_quantico_por_tempo_sqeuclidean.csv', index=False)

