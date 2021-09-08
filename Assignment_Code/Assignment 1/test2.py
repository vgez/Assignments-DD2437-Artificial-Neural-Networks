import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(100)
np_hist = np.random.normal(loc=0, scale=1, size=1000)

hist, bin_edges = np.histogram(np_hist)
print(hist)
print(bin_edges)
bin_edges = np.round(bin_edges, 0)
plt.figure(figsize=[10, 8])

plt.bar(bin_edges[:-1], hist, width=0.5, color='#0504aa', alpha=0.7)
plt.xlim(min(bin_edges), max(bin_edges))
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.title('Normal Distribution Histogram', fontsize=15)
plt.show()
