# Check ABBA is available.
import importlib
import warnings
spec = importlib.util.find_spec("ABBA")
if spec is None:
    warnings.warn("Try: pip install -r 'requirements.txt'")
from ABBA import ABBA as ABBA
import numpy as np


class forecaster(object):
    """
    Master class to deal with multiple forecasting techniques
    """

    def __init__(self, time_series, model, abba=None):
        self.model_class = model
        self.abba = abba
        self.time_series = time_series

        # Normalise time series
        self.mean = np.mean(self.time_series)
        self.std = np.std(self.time_series)
        self.normalised_time_series = (self.time_series - self.mean)/self.std if self.std!=0 else self.time_series - self.mean

        # Check if ABBA being used?
        if isinstance(self.abba, ABBA):
            self.pieces = abba.compress(self.normalised_time_series)
            self.ABBA_string, self.centers = self.abba.digitize(self.pieces)
            # Apply inverse transform.
            self.ABBA_numeric = self.mean + np.dot(self.std, self.abba.inverse_transform(self.ABBA_string, self.centers, self.normalised_time_series[0]))

            # One hot encode symbolic representation. Create list of all symbols
            # in case symbol does not occur in symbolic representation. (example:
            # flat line and insist k>1)
            self.alphabet = sorted([chr(97+i) for i in range(len(self.centers))])
            self.sequence = np.array([[0 if char != letter else 1 for char in self.alphabet] for letter in self.ABBA_string])
        else:
            self.sequence = self.normalised_time_series

        # Build model
        self.model_class.build(self.sequence)

        # Construct training data
        self.model_class.construct_training_index()

    def train(self, patience=100, max_epoch=100000, acceptable_loss=np.inf):
        self.model_class.train(patience=patience, max_epoch=max_epoch, acceptable_loss=acceptable_loss)

    def forecast(self, k):
        prediction = self.model_class.forecast(k)

        # Check if ABBA being used?
        if isinstance(self.abba, ABBA):
            s = ''
            for piece in prediction[-k:]:
                idx = np.argmax(piece, axis = 0)
                s += self.alphabet[idx]

            patches = self.abba.get_patches(self.normalised_time_series, self.pieces, self.ABBA_string, self.centers)
            # Construct mean of each patch
            d = {}
            for key in patches:
                d[key] = list(np.mean(patches[key], axis=0))

            prediction = [self.normalised_time_series[-1]]
            for letter in s:
                patch = d[letter]
                patch -= patch[0] - prediction[-1] # shift vertically
                prediction = prediction + patch[1:].tolist()
            if self.std == 0:
                return np.array(prediction[1:k+1]) + self.mean
            else:
                return np.array(prediction[1:k+1])*self.std + self.mean
        else:
            if self.std == 0:
                return np.array(prediction[len(self.normalised_time_series):]) + self.mean
            else:
                return np.array(prediction[len(self.normalised_time_series):])*self.std + self.mean
