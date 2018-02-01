import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
from past.builtins import xrange


warnings.filterwarnings("ignore")

    
def rmse(predictions, targets):
    return numpy.sqrt(((predictions - targets) ** 2).mean())

  
    