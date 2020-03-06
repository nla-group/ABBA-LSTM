import numpy as np
import sys
sys.path.append('./../../')
from forecaster import forecaster
from VanillaLSTM_keras import VanillaLSTM_keras
from ABBA import ABBA
from util import myfigure
from util import dtw as DTW
import itertools
import matplotlib.pyplot as plt
import itertools
import csv

def calc_dtw_error(prediction, n):
    n = int(n)
    ts2 = np.sin(np.linspace(-np.pi/2,(1199/999)*((2*np.pi*n))-np.pi/2, 1200))
    error = DTW(prediction, ts2[1000:])
    return error

def calc_euclid_error(prediction, n):
    n = int(n)
    ts2 = np.sin(np.linspace(-np.pi/2,(1199/999)*((2*np.pi*n))-np.pi/2, 1200))
    error = np.linalg.norm(prediction - ts2[1000:])
    return error

if __name__ == "__main__":
    stateful = False

    possible_n = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
    34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70,
    72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100]

    seed = [0, 1, 2, 3, 4]
    #seed = [5, 6, 7, 8, 9]

    inputs = list(itertools.product(possible_n, seed))

    for (n, seed) in inputs:
        print(n, seed)
        x = np.linspace(-np.pi/2,(2*np.pi*n)-np.pi/2, 1000)
        ts = np.sin(x)

        #abba = ABBA(verbose=0) # l=5 as using ABBA
        abba = None

        f = forecaster(ts, model=batchless_VanillaLSTM_pytorch(lag=5, stateful=stateful, seed=seed), abba=abba)
        f.train(patience=50, max_epoch=10000)
        prediction = f.forecast(200).tolist()

        row = [str(n), str(seed), str(calc_dtw_error(prediction, n)), str(calc_euclid_error(prediction, n))]

        if stateful:
            with open('stateful_results.csv', 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()
        else:
            with open('stateless_results.csv', 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()
