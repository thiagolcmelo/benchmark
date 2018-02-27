
# coding: utf-8

# In[72]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

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


# In[33]:


pe = pd.read_csv('onda_plana_pseudo_analitica_resultados_pe.csv')
rk = pd.read_csv('onda_plana_pseudo_analitica_resultados_rk.csv')
cn = pd.read_csv('onda_plana_pseudo_analitica_resultados_cn.csv')


# In[38]:


pec = pe.copy()
rkc = rk.copy()
cnc = cn.copy()

scaler = StandardScaler()
cols = ['stdvar', 'skew', 'a', 'stdvar_real', 'skew_real', 'a_real']

pec[cols] = scaler.fit_transform(pec[cols])
rkc[cols] = scaler.fit_transform(rkc[cols])
cnc[cols] = scaler.fit_transform(cnc[cols])

# In[73]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True);

with pd.plotting.plot_params.use('x_compat', True):
    pec['minkowski'] = pec.apply(lambda l: cdist(XA=[[l.a,l['stdvar'],l['skew']]], XB=[[l.a_real,l.stdvar_real,l.skew_real]], metric='minkowski', p=3)[0][0], axis=1)
    pec.plot(x='program_time', y='minkowski', kind='scatter', loglog=True, color='r', ax=ax1, sharex=True, sharey=True)

    rkc['minkowski'] = rkc.apply(lambda l: cdist(XA=[[l.a,l['stdvar'],l['skew']]], XB=[[l.a_real,l.stdvar_real,l.skew_real]], metric='minkowski', p=3)[0][0], axis=1)
    rkc.plot(x='program_time', y='minkowski', kind='scatter', loglog=True, color='g', ax=ax2, sharex=True, sharey=True)

    cnc['minkowski'] = cnc.apply(lambda l: cdist(XA=[[l.a,l['stdvar'],l['skew']]], XB=[[l.a_real,l.stdvar_real,l.skew_real]], metric='minkowski', p=3)[0][0], axis=1)
    cnc.plot(x='program_time', y='minkowski', kind='scatter', loglog=True, color='b', ax=ax3, sharex=True, sharey=True)

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
