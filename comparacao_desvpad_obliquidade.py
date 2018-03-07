import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, skewnorm
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

f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey=True)

mean, var, skew, kurt = norm.stats(moments='mvsk')

x = np.linspace(-10, 10, 1000)




ax1.plot(x, norm.pdf(x, loc=0, scale=0.5), 'r-', alpha=0.6, label='norm pdf')
ax1.text(-8, 0.6, r"$\sigma < \sigma_0$")
ax2.plot(x, norm.pdf(x, loc=0, scale=2.0), 'r-', alpha=0.6, label='norm pdf')
ax2.text(-8, 0.6, r"$\sigma = \sigma_0$")
ax3.plot(x, norm.pdf(x, loc=0, scale=4.0), 'r-', alpha=0.6, label='norm pdf')
ax3.text(-8, 0.6, r"$\sigma > \sigma_0$")

ax4.plot(x, skewnorm.pdf(x, -2, loc=0, scale=2.0), 'r-', alpha=0.6, label='norm pdf')
ax4.text(-8, 0.6, r"$\gamma < 0$")
ax5.plot(x, skewnorm.pdf(x, 0, loc=0, scale=2.0), 'r-', alpha=0.6, label='norm pdf')
ax5.text(-8, 0.6, r"$\gamma = 0$")
ax6.plot(x, skewnorm.pdf(x, 2, loc=0, scale=2.0), 'r-', alpha=0.6, label='norm pdf')
ax6.text(-8, 0.6, r"$\gamma > 0$")

plt.show()
