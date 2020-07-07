import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
import json

import ml_tools
import graphs

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

with open('lstm_config.json') as config_file:
    conf = json.load(config_file)


features = ['ENERGY', 'WS1', 'TEMP1', 'IRRAD1']

data = pd.read_excel('full_data.xlsx', sheet_name='data')
data['DateTime'] = pd.to_datetime(data['DateTime'])

data.set_index('DateTime', inplace=True)
data = data.astype(float)
data = data[features]
out_var = 'ENERGY'
data.plot(subplots = True)
#data_u = np.concatenate((data[out_var].values,np.flipud(data[out_var].values)))#duplicado y flipiado
dataset = data.values


train_split= ml_tools.data_split(dataset, 90)
#Normalize Just Train
dataset, data_mean, data_std = ml_tools.normaize(dataset)


past_hist= 120
future_target = 1
STEP = 1


x_train, y_train = ml_tools.multivariate_data(dataset, dataset[:, 0], 0,
                                                 train_split, past_hist,
                                                 future_target, STEP)
x_val, y_val = ml_tools.multivariate_data(dataset, dataset[:, 0],
                                             train_split, None, past_hist,
                                             future_target, STEP)

print ('Single window of past history : {}'.format(x_train[0].shape))
print ('\n Target temperature to predict : {}'.format(y_train[0].shape))

#graphs.multi_step_plot(x_train[0], y_train[0], np.array([0]), STEP)

##############################################################################################
model = Sequential()
model.add(LSTM(120, return_sequences=True, input_shape=x_train.shape[-2:]))
model.add(LSTM(16, activation ='relu'))
model.add(Dense(1))
print(model.summary())
model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae',metrics=[conf["metrics"]])
m_performance = model.fit(x_train, y_train, batch_size = 256, epochs= 10, shuffle = False, validation_data = (x_val, y_val))

graphs.plot_model_metric(m_performance, 'loss')

yhat = model.predict(x_val)

it = 0
graphs.multi_step_plot(x_val[it], y_val[it], yhat[it], STEP)

yhat = ml_tools.desnormalize(yhat, data_mean[0], data_std[0])

yhat = ml_tools.model_out_tunep(yhat)

graphs.plot_model_learn(data, yhat)
###############################################################################################




#n_ahead = 24
#last_input = 
#yhat2 = ml_tools.predict_n_ahead(model, n_ahead, last_input)
#graphs.plot_next_forecast(data, yhat, n_ahead)

