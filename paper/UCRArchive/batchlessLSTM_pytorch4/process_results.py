import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import myfigure
import os

colors=['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
# function for setting the colors of the box plots pairs
def setBoxColors(bp, colors):
    plt.setp(bp['boxes'][0], color=colors[0])
    plt.setp(bp['caps'][0], color=colors[0])
    plt.setp(bp['caps'][1], color=colors[0])
    plt.setp(bp['whiskers'][0], color=colors[0])
    plt.setp(bp['whiskers'][1], color=colors[0])
    plt.setp(bp['fliers'][0], color=colors[0])
    plt.setp(bp['fliers'][1], color=colors[0])
    plt.setp(bp['medians'][0], color=colors[0])

    plt.setp(bp['boxes'][1], color=colors[1])
    plt.setp(bp['caps'][2], color=colors[1])
    plt.setp(bp['caps'][3], color=colors[1])
    plt.setp(bp['whiskers'][2], color=colors[1])
    plt.setp(bp['whiskers'][3], color=colors[1])
    #plt.setp(bp['fliers'][2], color=colors[1])
    #plt.setp(bp['fliers'][3], color=colors[1])
    plt.setp(bp['medians'][1], color=colors[1])


ABBA_LSTM = pd.read_csv("ABBA_LSTM_results10.csv", sep = " ")
LSTM = pd.read_csv("LSTM_results10.csv", sep = " ")

# Remove results.csv if already exists.
try:
    os.remove('results.csv')
except:
    pass

# Write results to csv
with open('results.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([' ', 'LSTM', 'LSTM_ABBA'])

    for i, v in enumerate(['Time', 'sMAPE', 'Euclidean', 'diff_Euclidean', 'DTW', 'diff_DTW']):
        writer.writerow([v, np.mean(LSTM[v]), np.mean(ABBA_LSTM[v])])


# Distances boxplot
fig, (ax1) = myfigure(nrows=1, ncols=1, fig_ratio=0.5, fig_scale=1/0.6)
bp = ax1.boxplot([list(LSTM['Euclidean']), list(ABBA_LSTM['Euclidean'])], positions = [1, 2], widths = 0.6, whis='range')
setBoxColors(bp, colors)
bp = ax1.boxplot([list(np.sqrt(LSTM['DTW'])), list(np.sqrt(ABBA_LSTM['DTW']))], positions = [3, 4], widths = 0.6, whis='range')
setBoxColors(bp, colors)
bp = ax1.boxplot([list(LSTM['diff_Euclidean']), list(ABBA_LSTM['diff_Euclidean'])], positions = [5, 6], widths = 0.6, whis='range')
setBoxColors(bp, colors)
bp = ax1.boxplot([list(np.sqrt(LSTM['diff_DTW'])), list(np.sqrt(ABBA_LSTM['diff_DTW']))], positions = [7, 8], widths = 0.6, whis='range')
setBoxColors(bp, colors)
ax1.set_xticklabels(['Euclid', 'DTW', 'diff_Euclid', 'diff_DTW'])
ax1.set_xticks([1.5, 3.5, 5.5, 7.5])
hB, = ax1.plot([1,1], color=colors[0])
hR, = ax1.plot([1,1], color=colors[1])
leg = ax1.legend((hB, hR),('LSTM', 'ABBA_LSTM'), loc='lower center', bbox_to_anchor=(0.5, 0.95), ncol=2, framealpha=0)
leg.get_frame().set_linewidth(0.0)
hB.set_visible(False)
hR.set_visible(False)
plt.savefig('distances.pdf', dpi=300, transparent=True)


fig, (ax1) = myfigure(nrows=1, ncols=1, fig_ratio=0.5, fig_scale=1/0.6)
ind = np.arange(68)
width = 0.35
l1 = LSTM['Time'].to_numpy().ravel()
I = np.argsort(l1)
l2 = ABBA_LSTM['Time'].to_numpy().ravel()
b1 = ax1.bar(ind, l1[I], width, color=colors[0])
b2 = ax1.bar(ind+width, l2[I], width, color=colors[1])
ax1.legend((b1, b2), ('LSTM', 'ABBA_LSTM'))
ax1.set_xticklabels(['Time'])
plt.savefig('time.pdf', dpi=300, transparent=True)
