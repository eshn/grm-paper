import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('sample.csv')

lower = 500
upper = 700

time = df['Elapsed Time']
data = df['Glucose (mmol/L)']

plt.plot(time[lower:upper], data[lower:upper])
plt.grid(alpha=0.4)
plt.show()