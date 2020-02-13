from forecaster import forecaster
from VanillaLSTM_pytorch import VanillaLSTM_pytorch
from ABBA import ABBA as ABBA
import numpy as np

abba = ABBA()
model = VanillaLSTM_pytorch()

time_series = np.random.rand(100)

f = forecaster(time_series, model, abba=None)
f.train()
