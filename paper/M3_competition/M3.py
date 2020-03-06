import pandas as pd
import numpy as np
import sys
sys.path.append('./../../')
from forecaster import forecaster
from VanillaLSTM_keras import VanillaLSTM_keras
from ABBA import ABBA
import csv
from util import myfigure
from util import dtw as DTW
import matplotlib.pyplot as plt
import os
import time


# Import excel document into pandas dataframe
xls = pd.ExcelFile('M3C.xls')

# Select time series which are sampled monthly, should run from N1402 - N2829
Monthly = xls.parse(2)


def sMAPE(A, F):
    return 100/len(A) * np.sum(2 * np.abs(A - F) / (np.abs(A) + np.abs(F)))

lag = 10
patience = 100
fcast_len = 18

for index, row in Monthly[0:].iterrows():
    try:
        # import row and remove NaN padding
        ts = row.array[6:].to_numpy(dtype=np.float64)
        ts = ts[~np.isnan(ts)]

        if row['Series'] != 'N1500':
            raise Exception()

        train = ts[:-fcast_len]
        test = ts[-fcast_len:]

        # Build ABBA constructor
        abba = ABBA(tol=0.05, max_k = 10, verbose=0)

        # LSTM model with ABBA
        t0 = time.time()
        f = forecaster(ts, model=VanillaLSTM_keras(lag=lag), abba=abba)
        f.train(patience=patience, max_epoch=10000+patience)
        forecast1 = f.forecast(len(test)).tolist()
        t1 = time.time()

        # LSTM model without ABBA
        t0 = time.time()
        f = forecaster(time_series, model=VanillaLSTM_keras(lag=lag), abba=None)
        f.train(patience=patience, max_epoch=10000+patience)
        forecas21 = f.forecast(len(test)).tolist()
        t1 = time.time()

        ################################################################################
        #                   PLOTTING
        ################################################################################

        colors=['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
        # Produce plot
        fig, (ax1) = myfigure(nrows=1, ncols=1, fig_ratio=0.5, fig_scale=1/0.6)
        plt.subplots_adjust(left=0.125, bottom=None, right=0.95, top=None, wspace=None, hspace=None)
        ax1.plot(train, 'k')
        ax1.plot(model1.ABBA_representation_numerical, 'k', alpha=0.5)
        ax1.plot(range(len(train)-1, len(train)+len(test)), [train[-1]] + list(forecast1), color=colors[1], label='ABBA_LSTM')
        ax1.plot(range(len(train)-1, len(train)+len(test)), [train[-1]] + list(forecast2), color=colors[0], label='LSTM')
        plt.yticks([], [])

        # THETA
        Forecast = pd.read_excel('M3Forecast.xls', sheet_name='THETA', index_col=0)
        Forecast.columns=['N', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']
        fcast = Forecast.loc[row['Series'], :].to_numpy()[1:]
        fcast = np.hstack((ts[-19], fcast))
        plt.plot(range(len(train)-1, len(train)+len(test)), fcast, 'g', label='Theta')

        ax1.plot(range(len(train)-1, len(train)+len(test)), [train[-1]] + list(test), color=colors[2], label='truth')
        plt.legend(loc='lower center', bbox_to_anchor=(0.45, 0.95), ncol=4, framealpha=0)
        plt.savefig('./'+row['Series']+'.pdf')
        plt.close()

    except Exception as e:
        print(e)
