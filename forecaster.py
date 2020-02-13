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
            self.sequence = [[0 if char != letter else 1 for char in self.alphabet] for letter in self.ABBA_string]
        else:
            self.sequence = self.normalised_time_series

        # Build model
        self.model_class.build(self.sequence)

        # Construct training data
        self.model_class.construct_training_data()

    def train(self):
        self.model_class.train()

    def forecast(self, k):
        prediction = self.model_class.forecast(k)

        # must convert back?
