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
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU



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

def get_optimizer(conf):
	if conf['lr'] == None:
		if conf['optimizer'] == 'Adam':
			opt = tf.keras.optimizers.Adam()
		elif conf['optimizer'] == 'SGD':
			opt = tf.keras.optimizers.SGD()
		elif conf['optimizer'] == 'RMSprop':
			opt = tf.keras.optimizers.RMSprop()
		elif conf['optimizer'] == 'Adadelta':
			opt = tf.keras.optimizers.Adadelta()
		elif conf['optimizer'] == 'Adagrad':
			opt = tf.keras.optimizers.Adagrad()
		elif conf['optimizer'] == 'Adamax':
			opt = tf.keras.optimizers.Adamax()
		elif conf['optimizer'] == 'Nadam':
			opt = tf.keras.optimizers.Nadam()
		elif conf['optimizer'] == 'Ftrl':
			opt = tf.keras.optimizers.Ftrl()
	else:
		if conf['optimizer'] == 'Adam':
			opt = tf.keras.optimizers.Adam(learning_rate=conf["lr"])
		elif conf['optimizer'] == 'SGD':
			opt = tf.keras.optimizers.SGD(learning_rate=conf["lr"])
		elif conf['optimizer'] == 'RMSprop':
			opt = tf.keras.optimizers.RMSprop(learning_rate=conf["lr"])
		elif conf['optimizer'] == 'Adadelta':
			opt = tf.keras.optimizers.Adadelta(learning_rate=conf["lr"])
		elif conf['optimizer'] == 'Adagrad':
			opt = tf.keras.optimizers.Adagrad(learning_rate=conf["lr"])
		elif conf['optimizer'] == 'Adamax':
			opt = tf.keras.optimizers.Adamax(learning_rate=conf["lr"])
		elif conf['optimizer'] == 'Nadam':
			opt = tf.keras.optimizers.Nadam(learning_rate=conf["lr"])
		elif conf['optimizer'] == 'Ftrl':
			opt = tf.keras.optimizers.Ftrl(learning_rate=conf["lr"])
	return opt


def model_maker_Custom1(conf, x_train, y_train, x_val=[], y_val=[]):
	print("Custom1 Model")
	print("=============")
	model = Sequential()
	m_perf = {}
	if conf["layer2"] == 0:
	    model.add(LSTM(conf["layer1"], activation =conf["act_func"], input_shape=x_train.shape[-2:]))
	    #model.add(Dropout(conf["dropout"]))
	elif conf["layer3"] == 0:
	    #model.add(Bidirectional(LSTM(conf["layer1"], activation =conf["act_func"], input_shape=x_train.shape[-2:])))
	    # model.add(LSTM(conf["layer1"], activation =conf["act_func"], input_shape=x_train.shape[-2:]))
	    # #kernel_regularizer=tf.keras.regularizers.l2(conf["l2_reg"])
	    # #model.add(Dropout(conf["dropout"]))
	    # model.add(Dense(conf["layer2"], activation =conf["act_func"]))
	    #model.add(Dropout(conf["dropout"]))
	    #-------------------------------------------------
	    model.add(LSTM(conf["layer1"], input_shape=x_train.shape[-2:]))
	    model.add(LeakyReLU(alpha=0.1))
	    model.add(Dense(conf["layer2"]))
	    model.add(LeakyReLU(alpha=0.1))
	else:
	    model.add(LSTM(conf["layer1"], activation =conf["act_func"], input_shape=x_train.shape[-2:]))
	    model.add(Dropout(conf["dropout"]))
	    model.add(Dense(conf["layer2"], activation =conf["act_func"]))
	    model.add(Dropout(conf["dropout"]))
	    model.add(Dense(conf["layer3"], activation =conf["act_func"]))
	    model.add(Dropout(conf["dropout"]))
	if conf['future_target'] > 1:
		model.add(Dense(conf['future_target']))
	else:
		model.add(Dense(1))
	print(model.summary())
	if conf["callbacks"] == 1:
	    print("using callbacks...") #tensorboard --logdir=logs/fit
	    log_dir = settings.tb_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
	    early_s = EarlyStopping('loss', patience = conf["early_s"], mode = 'min', baseline = conf["baseline"])
	    lr_red = ReduceLROnPlateau('loss', patince= conf["early_s"], mode = 'min', verbose= conf["early_s"])
	    model.compile(optimizer=get_optimizer(conf), loss=conf["loss"],metrics=[conf["metrics"]+ [rmse]])
	    if x_val == [] or y_val == []:
	    	m_perf = model.fit(x_train, y_train, batch_size = conf["batch_size"], epochs= conf["epoch"], shuffle = False, callbacks=[early_s, tensorboard])
	    else:
	    	m_perf = model.fit(x_train, y_train, batch_size = conf["batch_size"], epochs= conf["epoch"], shuffle = False, validation_data = (x_val, y_val), callbacks=[early_s, lr_red, tensorboard])
	else:
	    model.compile(optimizer=get_optimizer(conf), loss=conf["loss"] ,metrics=[conf["metrics"] + [rmse]])
	    if x_val == [] or y_val == []:
	    	m_perf = model.fit(x_train, y_train, batch_size = conf["batch_size"], epochs= conf["epoch"], shuffle = False)
	    else:
	    	m_perf = model.fit(x_train, y_train, batch_size = conf["batch_size"], epochs= conf["epoch"], shuffle = False, validation_data = (x_val, y_val))
	return model, m_perf