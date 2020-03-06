from util import myfigure
from numpy import genfromtxt
import matplotlib.pyplot as plt

colors=['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']

# Scatter plots
data = genfromtxt('stateful_numeric.csv', delimiter=',')
possible_n = [2*i for i in range(1,51)]

# DTW
fig, (ax1) = myfigure(nrows=1, ncols=1, fig_ratio=0.71, fig_scale=1/0.6)
for n in possible_n:
    ax1.plot([n]*5, data[data[:,0] == n, 2]**(0.5), 'x', color=colors[0])
ax1.set_xlabel('n')
ax1.set_ylabel('dtw distance')
plt.xlim([0, 100])
plt.ylim([0.01, 1000])
plt.yscale('log')
plt.savefig('stateful_numeric_dtw.pdf', dpi=300, transparent=True)
plt.close()

# Euclidean
fig, (ax1) = myfigure(nrows=1, ncols=1, fig_ratio=0.71, fig_scale=1/0.6)
for n in possible_n:
    ax1.plot([n]*5, data[data[:,0] == n, 3], 'x', color=colors[0])
ax1.set_xlabel('n')
ax1.set_ylabel('Euclidean distance')
plt.xlim([0, 100])
plt.ylim([0.01, 1000])
plt.yscale('log')
plt.savefig('stateful_symbolic_euclidean.pdf', dpi=300, transparent=True)
plt.close()
