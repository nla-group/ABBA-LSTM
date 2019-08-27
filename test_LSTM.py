import unittest
import warnings
import numpy as np
import sys
from ABBA import ABBA as ABBA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from LSTM_keras import LSTM_model as LSTM_model_keras
from LSTM_pytorch import LSTM_model as LSTM_model_pytorch

def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test

# Ignore future warnings caused by tensorflow and numpy
warnings.simplefilter(action='ignore', category=FutureWarning)

class test_LSTM(unittest.TestCase):
    #--------------------------------------------------------------------------#
    # Check all numeric configurations
    #--------------------------------------------------------------------------#
    @ignore_warnings
    def test_Stateful_Numeric_Tb1_Keras(self):
        ts = [-2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2]
        model = LSTM_model_keras(num_layers=2, cells_per_layer=10, seed=0, dropout=0)
        model.build(ts, l=1, stateful=True, abba=None, verbose=False)
        model.train(patience=200, max_epoch=1000, verbose=False)
        model.end_prediction(20)
        prediction = model.end_prediction_ts[len(ts):]
        prediction = [round(p) for p in prediction]
        true_prediction = [-1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2]
        self.assertTrue(prediction == true_prediction)

    @ignore_warnings
    def test_Stateful_Numeric_Tb3_Keras(self):
        ts = [-2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2]
        model = LSTM_model_keras(num_layers=2, cells_per_layer=10, seed=0, dropout=0)
        model.build(ts, l=3, stateful=True, abba=None, verbose=False)
        model.train(patience=200, max_epoch=1000, verbose=False)
        model.end_prediction(20)
        prediction = model.end_prediction_ts[len(ts):]
        prediction = [round(p) for p in prediction]
        true_prediction = [-1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2]
        self.assertTrue(prediction == true_prediction)

    @ignore_warnings
    def test_Stateless_Numeric_Tb1_Keras(self):
        ts = [1]*30
        model = LSTM_model_keras(num_layers=2, cells_per_layer=10, seed=0, dropout=0)
        model.build(ts, l=1, stateful=False, abba=None, verbose=False)
        model.train(patience=200, max_epoch=1000, verbose=False)
        model.end_prediction(20)
        prediction = model.end_prediction_ts[len(ts):]
        prediction = [round(p) for p in prediction]
        true_prediction = [1]*20
        self.assertTrue(prediction == true_prediction)

    @ignore_warnings
    def test_Stateless_Numeric_Tb3_Keras(self):
        ts = [-2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2]
        model = LSTM_model_keras(num_layers=2, cells_per_layer=10, seed=0, dropout=0)
        model.build(ts, l=3, stateful=False, abba=None, verbose=False)
        model.train(patience=200, max_epoch=1000, verbose=False)
        model.end_prediction(20)
        prediction = model.end_prediction_ts[len(ts):]
        prediction = [round(p) for p in prediction]
        true_prediction = [-1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2]
        self.assertTrue(prediction == true_prediction)

    @ignore_warnings
    def test_Stateful_Numeric_Tb1_Pytorch(self):
        ts = [-2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2]
        model = LSTM_model_pytorch(num_layers=2, cells_per_layer=10, seed=0, dropout=0)
        model.build(ts, l=1, stateful=True, abba=None, verbose=False)
        model.train(patience=200, max_epoch=1000, verbose=False)
        model.end_prediction(20)
        prediction = model.end_prediction_ts[len(ts):]
        prediction = [round(p) for p in prediction]
        true_prediction = [-1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2]
        self.assertTrue(prediction == true_prediction)

    @ignore_warnings
    def test_Stateful_Numeric_Tb3_Pytorch(self):
        ts = [-2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2]
        model = LSTM_model_pytorch(num_layers=2, cells_per_layer=10, seed=0, dropout=0)
        model.build(ts, l=3, stateful=True, abba=None, verbose=False)
        model.train(patience=200, max_epoch=1000, verbose=False)
        model.end_prediction(20)
        prediction = model.end_prediction_ts[len(ts):]
        prediction = [round(p) for p in prediction]
        true_prediction = [-1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2]
        self.assertTrue(prediction == true_prediction)

    @ignore_warnings
    def test_Stateless_Numeric_Tb1_Pytorch(self):
        ts = [1]*30
        model = LSTM_model_pytorch(num_layers=2, cells_per_layer=10, seed=0, dropout=0)
        model.build(ts, l=1, stateful=False, abba=None, verbose=False)
        model.train(patience=200, max_epoch=1000, verbose=False)
        model.end_prediction(20)
        prediction = model.end_prediction_ts[len(ts):]
        prediction = [round(p) for p in prediction]
        true_prediction = [1]*20
        self.assertTrue(prediction == true_prediction)

    @ignore_warnings
    def test_Stateless_Numeric_Tb3_Pytorch(self):
        ts = [-2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2]
        model = LSTM_model_pytorch(num_layers=2, cells_per_layer=10, seed=0, dropout=0)
        model.build(ts, l=3, stateful=False, abba=None, verbose=False)
        model.train(patience=200, max_epoch=1000, verbose=False)
        model.end_prediction(20)
        prediction = model.end_prediction_ts[len(ts):]
        prediction = [round(p) for p in prediction]
        true_prediction = [-1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2]
        self.assertTrue(prediction == true_prediction)


    #--------------------------------------------------------------------------#
    # Check all symbolic configurations
    #--------------------------------------------------------------------------#
    @ignore_warnings
    def test_Stateful_Symbolic_Tb1_Keras(self):
        abba = ABBA(max_len=1, verbose=0)
        ts = [-2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2]
        model = LSTM_model_keras(num_layers=2, cells_per_layer=10, seed=0, dropout=0)
        model.build(ts, l=1, stateful=True, abba=abba, verbose=False)
        model.train(patience=200, max_epoch=1000, verbose=False)
        model.end_prediction(20)
        prediction = model.end_prediction_ts[len(ts):]
        prediction = [round(p) for p in prediction]
        true_prediction = [-1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2]
        self.assertTrue(prediction == true_prediction)

    @ignore_warnings
    def test_Stateful_Symbolic_Tb3_Keras(self):
        abba = ABBA(max_len=1, verbose=0)
        ts = [-2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2]
        model = LSTM_model_keras(num_layers=2, cells_per_layer=10, seed=0, dropout=0)
        model.build(ts, l=3, stateful=True, abba=abba, verbose=False)
        model.train(patience=200, max_epoch=1000, verbose=False)
        model.end_prediction(20)
        prediction = model.end_prediction_ts[len(ts):]
        prediction = [round(p) for p in prediction]
        true_prediction = [-1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2]
        self.assertTrue(prediction == true_prediction)

    @ignore_warnings
    def test_Stateless_Symbolic_Tb1_Keras(self):
        abba = ABBA(max_len=1, min_k=2, verbose=0)
        ts = [1]*30
        model = LSTM_model_keras(num_layers=2, cells_per_layer=10, seed=0, dropout=0)
        model.build(ts, l=1, stateful=False, abba=abba, verbose=False)
        model.train(patience=200, max_epoch=1000, verbose=False)
        model.end_prediction(20)
        prediction = model.end_prediction_ts[len(ts):]
        prediction = [round(p) for p in prediction]
        true_prediction = [1]*20
        self.assertTrue(prediction == true_prediction)

    @ignore_warnings
    def test_Stateless_Symbolic_Tb3_Keras(self):
        abba = ABBA(max_len=1, verbose=0)
        ts = [-2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2]
        model = LSTM_model_keras(num_layers=2, cells_per_layer=10, seed=0, dropout=0)
        model.build(ts, l=3, stateful=False, abba=abba, verbose=False)
        model.train(patience=200, max_epoch=1000, verbose=False)
        model.end_prediction(20)
        prediction = model.end_prediction_ts[len(ts):]
        prediction = [round(p) for p in prediction]
        true_prediction = [-3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22]
        self.assertTrue(prediction == true_prediction)

    @ignore_warnings
    def test_Stateful_Symbolic_Tb1_Pytorch(self):
        abba = ABBA(max_len=1, verbose=0)
        ts = [-2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2]
        model = LSTM_model_pytorch(num_layers=2, cells_per_layer=10, seed=1, dropout=0)
        model.build(ts, l=1, stateful=True, abba=abba, verbose=False)
        model.train(patience=200, max_epoch=1000, verbose=False)
        model.end_prediction(20)
        prediction = model.end_prediction_ts[len(ts):]
        prediction = [round(p) for p in prediction]
        true_prediction = [-1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2]
        self.assertTrue(prediction == true_prediction)

    @ignore_warnings
    def test_Stateful_Symbolic_Tb3_Pytorch(self):
        abba = ABBA(max_len=1, verbose=0)
        ts = [-2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2]
        model = LSTM_model_pytorch(num_layers=2, cells_per_layer=10, seed=0, dropout=0)
        model.build(ts, l=3, stateful=True, abba=abba, verbose=False)
        model.train(patience=200, max_epoch=1000, verbose=False)
        model.end_prediction(20)
        prediction = model.end_prediction_ts[len(ts):]
        prediction = [round(p) for p in prediction]
        true_prediction = [-1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2]
        self.assertTrue(prediction == true_prediction)

    @ignore_warnings
    def test_Stateless_Symbolic_Tb1_Pytorch(self):
        abba = ABBA(max_len=1, verbose=0)
        ts = [1]*30
        model = LSTM_model_pytorch(num_layers=2, cells_per_layer=10, seed=0, dropout=0)
        model.build(ts, l=1, stateful=False, abba=abba, verbose=False)
        model.train(patience=200, max_epoch=1000, verbose=False)
        model.end_prediction(20)
        prediction = model.end_prediction_ts[len(ts):]
        prediction = [round(p) for p in prediction]
        true_prediction = [1]*20
        self.assertTrue(prediction == true_prediction)

    @ignore_warnings
    def test_Stateless_Symbolic_Tb3_Pytorch(self):
        abba = ABBA(max_len=1, verbose=0)
        ts = [-2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1, 2, 1, 0, -1, -2]
        model = LSTM_model_pytorch(num_layers=2, cells_per_layer=10, seed=0, dropout=0)
        model.build(ts, l=3, stateful=False, abba=abba, verbose=False)
        model.train(patience=200, max_epoch=1000, verbose=False)
        model.end_prediction(20)
        prediction = model.end_prediction_ts[len(ts):]
        prediction = [round(p) for p in prediction]
        true_prediction = [-3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22]
        self.assertTrue(prediction == true_prediction)


if __name__ == "__main__":
    unittest.main()
