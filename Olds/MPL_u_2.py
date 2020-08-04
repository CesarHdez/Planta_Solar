import numpy as np
import pandas as pd
import json
import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
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
import graphs
import ml_tools

ml_tools.clean_output_folders()
#with open(settings.conf_path) as config_file:
with open(settings.conf_path) as config_file:
    conf = json.load(config_file)

data = pd.read_excel(settings.ex_data, sheet_name='data')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)
data = data.astype(float)
#data = data['ENERGY']
data[conf["y_var"]].astype(float)
#data.plot(subplots = True)
#data_u = np.concatenate((data[out_var].values,np.flipud(data[out_var].values)))#duplicado y flipiado
data_u = data[conf["y_var"]].values


data_u, data_mean, data_std = ml_tools.normaize(data_u)

train_split= ml_tools.data_split(data_u, conf["split_p"])

u_past_hist= conf["past_hist"]
u_future_traget = conf["future_target"]

x_train, y_train = ml_tools.univariate_data(data_u, 0, train_split, u_past_hist, u_future_traget)
x_val, y_val = ml_tools.univariate_data(data_u, train_split, None, u_past_hist, u_future_traget)

#x_train = np.reshape(x_train, (len(x_train), x_train.shape[1]))


model = Sequential()
model.add(Dense(100, activation='elu', input_shape=x_train.shape[-2:]))
model.add(Dropout(conf["dropout"]))
model.add(Dense(100, activation='elu'))
model.add(Flatten())
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
m_perf = model.fit(x_train, y_train, batch_size = conf["batch_size"], epochs= conf["epoch"], validation_data = (x_val, y_val))

yhat= model.predict(x_val)

yhat = ml_tools.desnormalize(yhat, data_mean, data_std)

yhat = ml_tools.model_out_tunep(yhat)

graphs.plot_model_learn(data, yhat, True)

















