import sys
sys.path.append('./../../')
from forecaster import forecaster
import numpy as np
from ABBA import ABBA
from util import myfigure
from batchless_VanillaLSTM_pytorch import batchless_VanillaLSTM_pytorch
import os
import csv
import matplotlib.pyplot as plt

datadir = './../../../../../ABBA/py/UCRArchive_2018'

def get_n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


lag = 10
patience = 50
fcast_len = 50

datasets = ['Earthquakes', 'Coffee', 'OSULeaf']
colors=['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']

for dataset in datasets:
    with open(datadir+'/'+dataset+'/'+dataset+'_TEST.tsv') as tsvfile:
        tsvfile = csv.reader(tsvfile, delimiter='\t')
        col = next(tsvfile)
        ts = [float(i) for i in col]

    # remove class information
    ts = np.array(ts[1:])
    # remove NaN from time series
    ts = ts[~np.isnan(ts)]

    # Normalise time series
    ts -= np.mean(ts)
    ts /= np.std(ts)

    train = ts[:-fcast_len]
    test = ts[-fcast_len:]

    # Build ABBA constructor
    abba = ABBA(tol=0.05, max_k=9, verbose=0)
    string, centers = abba.transform(train)
    abba_numerical = abba.inverse_transform(string, centers, train[0])

    # LSTM model with ABBA
    f = forecaster(train, model=batchless_VanillaLSTM_pytorch(lag=lag), abba=abba)
    print(get_n_params(f.model_class.model))
    f.train(patience=patience, max_epoch=10000+patience, batch_size=32)
    forecast1 = f.forecast(len(test)).tolist()

    # LSTM model without ABBA
    f = forecaster(train, model=batchless_VanillaLSTM_pytorch(lag=lag), abba=None)
    f.train(patience=patience, max_epoch=10000+patience, batch_size=32)
    print(get_n_params(f.model_class.model))
    forecast2 = f.forecast(len(test)).tolist()

    # Produce plot
    fig, (ax1) = myfigure(nrows=1, ncols=1, fig_ratio=0.5, fig_scale=1/0.6)
    plt.subplots_adjust(left=0.125, bottom=None, right=0.95, top=None, wspace=None, hspace=None)
    ax1.plot(train, 'k')
    ax1.plot(abba_numerical, 'k', alpha=0.5)
    ax1.plot(range(len(train)-1, len(train)+len(test)), [train[-1]] + list(forecast1), color=colors[1], label='ABBA_LSTM')
    ax1.plot(range(len(train)-1, len(train)+len(test)), [train[-1]] + list(forecast2), color=colors[0], label='LSTM')
    ax1.plot(range(len(train)-1, len(train)+len(test)), [train[-1]] + list(test), color=colors[2], label='truth')
    plt.legend(loc='lower center', bbox_to_anchor=(0.45, 0.95), ncol=3, framealpha=0)
    plt.savefig('./' + dataset + '.pdf')
    plt.close()
