from forecaster import forecaster
from VanillaLSTM_pytorch import VanillaLSTM_pytorch
from VanillaLSTM_keras import VanillaLSTM_keras
from ABBA import ABBA as ABBA
import numpy as np

time_series = [1, 2, 3, 2]*100 + [1]

k = 10

################################################################################
# Keras Vanilla LSTM
################################################################################
# stateful numeric
#f = forecaster(time_series, model=VanillaLSTM_keras(stateful=True), abba=None)
#f.train(max_epoch=100)
#print(f.forecast(k).tolist())

# stateless numeric
#f = forecaster(time_series, model=VanillaLSTM_keras(stateful=False), abba=None)
#f.train(max_epoch=100)
#print(f.forecast(k).tolist())

# stateful symbolic
#f = forecaster(time_series, model=VanillaLSTM_keras(stateful=True), abba=ABBA(max_len=2, verbose=0))
#f.train(max_epoch=100)
#print(f.forecast(k).tolist())

# stateless symbolic
#f = forecaster(time_series, model=VanillaLSTM_keras(stateful=False), abba=ABBA(max_len=2, verbose=0))
#f.train(max_epoch=100)
#print(f.forecast(k).tolist())

################################################################################
# Pytorch Vanilla LSTM
################################################################################
# stateful numeric
f = forecaster(time_series, model=VanillaLSTM_pytorch(stateful=True), abba=None)
f.train(max_epoch=100)
print(f.forecast(k).tolist())

# stateless numeric
f = forecaster(time_series, model=VanillaLSTM_pytorch(stateful=False), abba=None)
f.train(max_epoch=100)
print(f.forecast(k).tolist())

# stateful symbolic
f = forecaster(time_series, model=VanillaLSTM_pytorch(stateful=True), abba=ABBA(max_len=2, verbose=0))
f.train(max_epoch=100)
print(f.forecast(k).tolist())

# stateless symbolic
f = forecaster(time_series, model=VanillaLSTM_pytorch(stateful=False), abba=ABBA(max_len=2, verbose=0))
f.train(max_epoch=100)
print(f.forecast(k).tolist())
