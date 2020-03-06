from util import myfigure
from numpy import genfromtxt
import matplotlib.pyplot as plt

colors=['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']

# Calculate best can achieve due to ABBA representation
ABBA_rep = genfromtxt('./../symbolic_representations/ABBA_representation.csv', delimiter=',')
dtw_distance = ABBA_rep[:, 1]
euclid_distance = ABBA_rep[:, 2]

# Scatter plots
data = genfromtxt('stateless_symbolic.csv', delimiter=',')
possible_n = [2*i for i in range(1,51)]

# DTW
fig, (ax1) = myfigure(nrows=1, ncols=1, fig_ratio=0.71, fig_scale=1/0.6)
#ax1.plot(possible_n, dtw_distance, 'ko')
ax1.plot(possible_n, [2**(0.5)]*len(possible_n), 'k')
for n in possible_n:
    ax1.plot([n]*5, data[data[:,0] == n, 2]**(0.5), 'x', color=colors[1])
ax1.set_xlabel('n')
ax1.set_ylabel('dtw distance')
plt.xlim([0, 100])
plt.ylim([0.01, 1000])
plt.yscale('log')
plt.savefig('stateless_symbolic_dtw.pdf', dpi=300, transparent=True)
plt.close()

# Euclidean
fig, (ax1) = myfigure(nrows=1, ncols=1, fig_ratio=0.71, fig_scale=1/0.6)
#ax1.plot(possible_n, euclid_distance, 'ko')
ax1.plot(possible_n, [2**(0.5)]*len(possible_n), 'k')
for n in possible_n:
    ax1.plot([n]*5, data[data[:,0] == n, 3], 'x', color=colors[1])
ax1.set_xlabel('n')
ax1.set_ylabel('Euclidean distance')
plt.xlim([0, 100])
plt.ylim([0.01, 1000])
plt.yscale('log')
plt.savefig('stateless_symbolic_euclidean.pdf', dpi=300, transparent=True)
plt.close()
