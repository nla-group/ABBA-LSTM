import numpy as np
from ABBA import ABBA as ABBA
from util import dtw as DTW
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('classic')
import csv

possible_n = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
    34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70,
    72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100]

for n in possible_n:
    print(n)
    x = np.linspace(-np.pi/2,(2*np.pi*n)-np.pi/2, 1000)
    original_ts = np.sin(x)

    # Normalise
    mean = np.mean(original_ts)
    std = np.std(original_ts)
    ts = (original_ts - mean)/std if std!=0 else original_ts - mean

    abba = ABBA(verbose=0)
    pieces = abba.compress(ts)
    ABBA_representation_string, centers = abba.digitize(pieces)
    ABBA_representation_numerical = abba.inverse_transform(ABBA_representation_string, centers, ts[0])


    # Calculate best possible distance
    full_x =  np.sin(np.linspace(-np.pi/2,(1399/999)*((2*np.pi*n))-np.pi/2, 1400))
    pred_x = full_x[1000:]
    normalised_pred_x = (pred_x - mean)/std
    pred_pieces = abba.compress(normalised_pred_x)

    s = ''
    for [len, inc, error] in pred_pieces:
        cluster_inc = centers[:, 1]
        # Note scl = 0
        index = np.argmin(np.abs(cluster_inc - inc)) #find index of cluster
        s += chr(97 + index)


    # Patches
    ABBA_patches = abba.get_patches(ts, pieces, ABBA_representation_string, centers)
    # Construct mean of each patch
    d = {}
    for key in ABBA_patches:
        d[key] = list(np.mean(ABBA_patches[key], axis=0))

    # Stitch patches together
    patched_ts = np.array([normalised_pred_x[0]])
    for letter in s:
        patch = d[letter]
        patch -= patch[0] - patched_ts[-1] # shift vertically
        patched_ts = np.hstack((patched_ts, patch[1:]))
    pred_reconstructed =  mean + np.dot(std, patched_ts[1:])

    dtw_distance = DTW(pred_x[0:200], pred_reconstructed[0:200])**(0.5)
    euclid_distance = np.linalg.norm(pred_x[0:200] - pred_reconstructed[0:200])

    row = [str(n), str(dtw_distance), str(euclid_distance)]

    with open('ABBA_representation.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()

    fig_ratio = .5
    fig_scale = 1
    plt.figure(num=None, figsize=(5/fig_scale, 5*fig_ratio/fig_scale), dpi=80*fig_scale, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.11*fig_scale, right=1-0.05*fig_scale, bottom=0.085*fig_scale/fig_ratio, top=1-0.05*fig_scale/fig_ratio)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('DTW: ' + str(dtw_distance) + ' Euclidean: ' + str(euclid_distance))
    ax1.plot(original_ts, 'k', label='original')
    ax1.plot(np.array(ABBA_representation_numerical)*std + mean, 'r', label='ABBA')
    ax1.legend()

    inv_comp = np.array(abba.inverse_compress(normalised_pred_x[0], pred_pieces))*std + mean
    ax2.plot(pred_x[0:200], 'k', label='truth')
    ax2.plot(inv_comp[0:200], 'g', label='compressed')
    ax2.plot(pred_reconstructed[0:200], 'r', label='patched ABBA')
    ax2.axvspan(0, 200, color='red', alpha=0.2)
    ax2.legend()

    plt.savefig('phase_' + str(n) + '.pdf')
    plt.close('all')
