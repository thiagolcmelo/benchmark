import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
df = pd.read_csv('cn.csv')
sns.pairplot(df);
plt.show()
