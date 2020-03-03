import unittest
from forecaster import forecaster
from batchless_VanillaLSTM_pytorch import batchless_VanillaLSTM_pytorch
from batchless_VanillaLSTM_keras import batchless_VanillaLSTM_keras
from VanillaLSTM_keras import VanillaLSTM_keras
from ABBA import ABBA as ABBA
import numpy as np

class test_LSTM(unittest.TestCase):

    ##################################################
    # batchless_VanillaLSTM_keras
    ##################################################
    def test_VanillaLSTM_stateful_numeric_keras(self):
        time_series = [1, 2, 3, 2]*100 + [1]
        k = 10
        f = forecaster(time_series, model=batchless_VanillaLSTM_keras(stateful=True), abba=None)
        f.train(max_epoch=300, patience=20)
        prediction = f.forecast(k).tolist()
        prediction = [round(p) for p in prediction]
        print(prediction)
        self.assertTrue(prediction == [2, 3, 2, 1, 2, 3, 2, 1, 2, 3])

    def test_VanillaLSTM_stateless_numeric_keras(self):
        time_series = [1, 2, 3, 2]*100 + [1]
        k = 10
        f = forecaster(time_series, model=batchless_VanillaLSTM_keras(stateful=False), abba=None)
        f.train(max_epoch=300, patience=20)
        prediction = f.forecast(k).tolist()
        prediction = [round(p) for p in prediction]
        print(prediction)
        self.assertTrue(prediction == [2, 3, 2, 1, 2, 3, 2, 1, 2, 3])

    def test_VanillaLSTM_stateful_symbolic_keras(self):
        time_series = [1, 2, 3, 2]*100 + [1]
        k = 10
        f = forecaster(time_series, model=batchless_VanillaLSTM_keras(stateful=True), abba=ABBA(max_len=2, verbose=0))
        f.train(max_epoch=300, patience=20)
        prediction = f.forecast(k).tolist()
        prediction = [round(p) for p in prediction]
        print(prediction)
        self.assertTrue(prediction == [2, 3, 2, 1, 2, 3, 2, 1, 2, 3])

    def test_VanillaLSTM_stateless_symbolic_keras(self):
        time_series = [1, 2, 3, 2]*100 + [1]
        k = 10
        f = forecaster(time_series, model=batchless_VanillaLSTM_keras(stateful=False), abba=ABBA(max_len=2, verbose=0))
        f.train(max_epoch=300, patience=20)
        prediction = f.forecast(k).tolist()
        prediction = [round(p) for p in prediction]
        print(prediction)
        self.assertTrue(prediction == [2, 3, 2, 1, 2, 3, 2, 1, 2, 3])


    ##################################################
    # batchless_VanillaLSTM_pytorch
    ##################################################
    def test_VanillaLSTM_stateful_numeric_pytorch(self):
        time_series = [1, 2, 3, 2]*100 + [1]
        k = 10
        f = forecaster(time_series, model=batchless_VanillaLSTM_pytorch(stateful=True), abba=None)
        f.train(max_epoch=300, patience=20)
        prediction = f.forecast(k).tolist()
        prediction = [round(p) for p in prediction]
        print(prediction)
        self.assertTrue(prediction == [2, 3, 2, 1, 2, 3, 2, 1, 2, 3])

    def test_VanillaLSTM_stateless_numeric_pytorch(self):
        time_series = [1, 2, 3, 2]*100 + [1]
        k = 10
        f = forecaster(time_series, model=batchless_VanillaLSTM_pytorch(stateful=False), abba=None)
        f.train(max_epoch=300, patience=20)
        prediction = f.forecast(k).tolist()
        prediction = [round(p) for p in prediction]
        print(prediction)
        self.assertTrue(prediction == [2, 3, 2, 1, 2, 3, 2, 1, 2, 3])

    def test_VanillaLSTM_stateful_symbolic_pytorch(self):
        time_series = [1, 2, 3, 2]*100 + [1]
        k = 10
        f = forecaster(time_series, model=batchless_VanillaLSTM_pytorch(stateful=True), abba=ABBA(max_len=2, verbose=0))
        f.train(max_epoch=300, patience=20)
        prediction = f.forecast(k).tolist()
        prediction = [round(p) for p in prediction]
        print(prediction)
        self.assertTrue(prediction == [2, 3, 2, 1, 2, 3, 2, 1, 2, 3])

    def test_VanillaLSTM_stateless_symbolic_pytorch(self):
        time_series = [1, 2, 3, 2]*100 + [1]
        k = 10
        f = forecaster(time_series, model=batchless_VanillaLSTM_pytorch(stateful=False), abba=ABBA(max_len=2, verbose=0))
        f.train(max_epoch=300, patience=20)
        prediction = f.forecast(k).tolist()
        prediction = [round(p) for p in prediction]
        print(prediction)
        self.assertTrue(prediction == [2, 3, 2, 1, 2, 3, 2, 1, 2, 3])

    ##################################################
    # VanillaLSTM_keras
    ##################################################
    def test_VanillaLSTM_batch_numeric_keras(self):
        time_series = [1, 2, 3, 2]*100 + [1]
        k = 10
        f = forecaster(time_series, model=VanillaLSTM_keras(), abba=None)
        f.train(max_epoch=300, patience=20)
        prediction = f.forecast(k).tolist()
        prediction = [round(p) for p in prediction]
        print(prediction)
        self.assertTrue(prediction == [2, 3, 2, 1, 2, 3, 2, 1, 2, 3])

    def test_VanillaLSTM_batch_symbolic_keras(self):
        time_series = [1, 2, 3, 2]*100 + [1]
        k = 10
        f = forecaster(time_series, model=VanillaLSTM_keras(), abba=ABBA(max_len=2, verbose=0))
        f.train(max_epoch=300, patience=20)
        prediction = f.forecast(k).tolist()
        prediction = [round(p) for p in prediction]
        print(prediction)
        self.assertTrue(prediction == [2, 3, 2, 1, 2, 3, 2, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
