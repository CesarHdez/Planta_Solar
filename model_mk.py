import numpy as np
import pandas as pd
import json
import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dropout


from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf

#from keras.models import Sequential
#from keras.models import load_model
#from keras.layers import Dense
#from keras.layers import LSTM

import settings

from models_mk import SimpleRNN
from models_mk import MPL
from models_mk import LSTM
from models_mk import GRU
from models_mk import Custom1
from models_mk import Custom2

def model_maker(conf, x_train, y_train, x_val=[], y_val=[]):
	print("Model Maker Working...")
	if conf["type"] == "LSTM":
		model, m_perf = LSTM.model_maker_LSTM(conf, x_train, y_train, x_val, y_val)
	elif conf["type"] == "GRU":
		model, m_perf = GRU.model_maker_GRU(conf, x_train, y_train, x_val, y_val)
	elif conf["type"] == "SimpleRNN":
		model, m_perf = SimpleRNN.model_maker_SimpleRNN(conf, x_train, y_train, x_val, y_val)
	elif conf["type"] == "MPL":
		model, m_perf = MPL.model_maker_MPL(conf, x_train, y_train, x_val, y_val)
	elif conf["type"] == "Custom1":
		model, m_perf = Custom1.model_maker_Custom1(conf, x_train, y_train, x_val, y_val)
	elif conf["type"] == "Custom2":
		model, m_perf = Custom2.model_maker_Custom2(conf, x_train, y_train, x_val, y_val)	
	return model, m_perf
