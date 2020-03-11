import sys
sys.path.append('./../../../')
from forecaster import forecaster
import numpy as np
from ABBA import ABBA
from util import myfigure
from util import dtw as DTW
from batchless_VanillaLSTM_pytorch import batchless_VanillaLSTM_pytorch
import os
import csv
import time
import matplotlib.pyplot as plt

def sMAPE(A, F):
    return 100/len(A) * np.sum(2 * np.abs(A - F) / (np.abs(A) + np.abs(F)))


datadir = './../../../../../../ABBA/py/UCRArchive_2018'

lag = 10
patience = 100
fcast_len = 50

###############################################################################
# Add header to csv file

#header = ['Dataset', 'Length', 'Epoch', 'Loss', 'Time', 'sMAPE', 'Euclidean', 'diff_Euclidean', 'DTW', 'diff_DTW']
header = ['Dataset', 'Length', 'Time', 'sMAPE', 'Euclidean', 'diff_Euclidean', 'DTW', 'diff_DTW']
if not os.path.isfile('./ABBA_LSTM_results'+str(lag)+'.csv'):
    with open('./ABBA_LSTM_results'+str(lag)+'.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)

if not os.path.isfile('./LSTM_results'+str(lag)+'.csv'):
    with open('./LSTM_results'+str(lag)+'.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)

###############################################################################
# Run through dataset
for root, dirs, files in os.walk(datadir):
    if dirs != []:
        for dataset in dirs:
            try:
                print('Dataset:', dataset, 'Lag:', lag)

                if os.path.isfile('./plots/' + dataset + '_' + str(lag) + '.pdf'):
                    raise RuntimeError('Already complete.')

                # Import time series
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
                abba = ABBA(tol=0.05, max_k = 10, verbose=0)
                string, centers = abba.transform(train)
                abba_numerical = abba.inverse_transform(string, centers, train[0])

                if len(train) < 100:
                    raise RuntimeError('Time series too short')
                if len(string) < 20:
                    raise RuntimeError('Time series too short')

                # LSTM model with ABBA
                t0 = time.time()
                f = forecaster(train, model=batchless_VanillaLSTM_pytorch(lag=lag), abba=abba)
                f.train(patience=patience, max_epoch=10000+patience)
                forecast1 = f.forecast(len(test)).tolist()
                t1 = time.time()

                with open('./ABBA_LSTM_results'+str(lag)+'.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=' ',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    #epoch = len(model1.loss)-patience
                    #loss = model1.loss[epoch]
                    t = t1 - t0
                    smape = sMAPE(test, forecast1)
                    euclid = np.linalg.norm(test - forecast1)
                    diff_euclid = np.linalg.norm(np.diff(test)-np.diff(forecast1))
                    dtw = DTW(test, forecast1)
                    diff_dtw = DTW(np.diff(test), np.diff(forecast1))

                    #row = [dataset, len(string), epoch, loss, t, smape, euclid, diff_euclid, dtw, diff_dtw]
                    row = [dataset, len(string), t, smape, euclid, diff_euclid, dtw, diff_dtw]
                    writer.writerow(row)

                # LSTM model without ABBA
                t0 = time.time()
                f = forecaster(train, model=batchless_VanillaLSTM_pytorch(lag=lag), abba=None)
                f.train(patience=patience, max_epoch=10000+patience)
                forecast2 = f.forecast(len(test)).tolist()
                t1 = time.time()

                with open('./LSTM_results'+str(lag)+'.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=' ',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    #epoch = len(model2.loss)-patience
                    #loss = model2.loss[epoch]
                    t = t1 - t0
                    smape = sMAPE(test, forecast2)
                    euclid = np.linalg.norm(test - forecast2)
                    diff_euclid = np.linalg.norm(np.diff(test)-np.diff(forecast2))
                    dtw = DTW(test, forecast2)
                    diff_dtw = DTW(np.diff(test), np.diff(forecast2))

                    #row = [dataset, len(train), epoch, loss, t, smape, euclid, diff_euclid, dtw, diff_dtw]
                    row = [dataset, len(train), t, smape, euclid, diff_euclid, dtw, diff_dtw]
                    writer.writerow(row)

                # Produce plot
                fig, (ax1) = myfigure(nrows=1, ncols=1, fig_ratio=0.71, fig_scale=1)
                plt.subplots_adjust(left=0.125, bottom=None, right=0.95, top=None, wspace=None, hspace=None)
                ax1.plot(train, 'k')
                ax1.plot(abba_numerical, 'k', alpha=0.5, label='ABBA representation')
                ax1.plot(range(len(train)-1, len(train)+len(test)), [train[-1]] + list(forecast1), 'r', label='ABBA_LSTM')
                ax1.plot(range(len(train)-1, len(train)+len(test)), [train[-1]] + list(forecast2), 'b', label='LSTM')
                ax1.plot(range(len(train)-1, len(train)+len(test)), [train[-1]] + list(test), 'y', label='truth')
                plt.legend(loc=6)
                plt.savefig('./plots/' + dataset + '_' + str(lag) + '.pdf')
                plt.close()
            except Exception as e:
                print(e)
