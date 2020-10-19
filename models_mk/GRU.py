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

rmse = tf.keras.metrics.RootMeanSquaredError()


def model_maker_GRU(conf, x_train, y_train, x_val=[], y_val=[]):
	model = Sequential()
	m_perf = {}
	if conf["layer2"] == 0:
	    model.add(GRU(conf["layer1"], activation =conf["act_func"], input_shape=x_train.shape[-2:]))
	    model.add(Dropout(conf["dropout"]))
	elif conf["layer3"] == 0:
	    model.add(GRU(conf["layer1"], return_sequences=True, activation =conf["act_func"], input_shape=x_train.shape[-2:]))
	    #kernel_regularizer=tf.keras.regularizers.l2(conf["l2_reg"])
	    model.add(Dropout(conf["dropout"]))
	    model.add(GRU(conf["layer2"], activation =conf["act_func"]))
	    model.add(Dropout(conf["dropout"]))
	else:
	    model.add(GRU(conf["layer1"], return_sequences=True, activation =conf["act_func"], input_shape=x_train.shape[-2:]))
	    model.add(Dropout(conf["dropout"]))
	    model.add(GRU(conf["layer2"], return_sequences=True, activation =conf["act_func"]))
	    model.add(Dropout(conf["dropout"]))
	    model.add(GRU(conf["layer3"], activation =conf["act_func"]))
	    model.add(Dropout(conf["dropout"]))
	model.add(Dense(1))
	print(model.summary())
	if conf["callbacks"] == 1:
	    print("using callbacks...") #tensorboard --logdir=logs/fit
	    log_dir = settings.tb_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
	    early_s = EarlyStopping('loss', patience = conf["early_s"], mode = 'min')
	    lr_red = ReduceLROnPlateau('loss', patince= conf["early_s"], mode = 'min', verbose= conf["early_s"])
	    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=conf["lr"]), loss=conf["loss"],metrics=[conf["metrics"]+ [rmse]])
	    if x_val == [] or y_val == []:
	    	m_perf = model.fit(x_train, y_train, batch_size = conf["batch_size"], epochs= conf["epoch"], shuffle = False, callbacks=[early_s, lr_red, tensorboard])
	    else:
	    	m_perf = model.fit(x_train, y_train, batch_size = conf["batch_size"], epochs= conf["epoch"], shuffle = False, validation_data = (x_val, y_val), callbacks=[early_s, lr_red, tensorboard])
	else:
	    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=conf["lr"]), loss=conf["loss"],metrics=[conf["metrics"]+ [rmse]])
	    if x_val == [] or y_val == []:
	    	m_perf = model.fit(x_train, y_train, batch_size = conf["batch_size"], epochs= conf["epoch"], shuffle = False)
	    else:
	    	m_perf = model.fit(x_train, y_train, batch_size = conf["batch_size"], epochs= conf["epoch"], shuffle = False, validation_data = (x_val, y_val))
	return model, m_perf