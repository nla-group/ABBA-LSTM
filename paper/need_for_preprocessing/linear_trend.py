import numpy as np
import sys
sys.path.append('./../..')
from forecaster import forecaster
from VanillaLSTM_keras import VanillaLSTM_keras
import matplotlib as mpl
import matplotlib.pyplot as plt
from util import myfigure
from ABBA import ABBA

trend = np.linspace(0, 0.5, 100)

colors=['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']

end_numeric = []
for s in range(10):
    f = forecaster(trend, model=VanillaLSTM_keras(lag=20), abba=None)
    f.train(patience=10, max_epoch=100000)
    end_numeric.append(f.forecast(100).tolist())

fig, (ax1) = myfigure(nrows=1, ncols=1, fig_ratio=0.71, fig_scale=1/0.6)
ax1.plot(np.linspace(0, 0.5, 100), label='training')
for i in range(10):
    if i == 0:
        ax1.plot(np.arange(100,200), end_numeric[i], color=colors[1], linestyle='dashed', label='forecast')
    else:
        ax1.plot(np.arange(100,200), end_numeric[i], color=colors[1], linestyle='dashed')
ax1.plot(np.arange(100,200), np.linspace(0.5, 1, 100)[0:], color=colors[2], label='truth')
ax1.axis([0, 200, 0, 1])
ax1.legend(loc=2)
plt.savefig('linear_trend_numeric.pdf', dpi=300, transparent=True)

end_symbolic = []
abba = ABBA(max_len = 10)
for s in range(10):
    f = forecaster(trend, model=VanillaLSTM_keras(lag=5), abba=abba)
    f.train(patience=10, max_epoch=100000)
    end_symbolic.append(f.forecast(100).tolist())

fig, (ax2) = myfigure(nrows=1, ncols=1, fig_ratio=0.71, fig_scale=1/0.6)
ax2.plot(np.linspace(0, 0.5, 100), label='training')
for i in range(10):
    if i == 0:
        ax2.plot(np.arange(100,200), end_symbolic[i], color=colors[1], linestyle='dashed', label='forecast')
    else:
        ax2.plot(np.arange(100,200), end_symbolic[i], color=colors[1], linestyle='dashed')
ax2.plot(np.arange(100,200), np.linspace(0.5, 1, 100)[0:], color=colors[2], label='truth')
ax2.axis([0, 200, 0, 1])
ax2.legend(loc=2)
plt.savefig('linear_trend_symbolic.pdf', dpi=300, transparent=True)
