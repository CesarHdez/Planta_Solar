import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf

import ml_tools
import graphs

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


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


model = Sequential()
model.add(LSTM(120, return_sequences=True, input_shape=x_train.shape[-2:]))
model.add(LSTM(16, activation ='relu'))
model.add(Dense(1))
model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
m_performance = model.fit(x_train, y_train, batch_size = 256, epochs= 10, shuffle = False, validation_data = (x_val, y_val))

graphs.plot_model_metric(m_performance, 'loss')

yhat = model.predict(x_val)

it = 0
graphs.multi_step_plot(x_val[it], y_val[it], yhat[it], STEP)

#Constructing the forecast dataframe
#fc = data.tail(len(yhat)).copy()
##fc.reset_index(inplace=True)
#
#yhat = desnormalize(yhat[0], data_mean[0], data_std[0])
#
#fc['forecast'] = yhat
#fc = fc[[ 'ENERGY', 'forecast']]
## Ploting the forecasts
#plt.figure(figsize=(20, 8))
#for dtype in ['ENERGY', 'forecast']:
#    plt.plot(
#        fc.index,
#        dtype,
#        data=fc,
#        label=dtype,
#        alpha=0.8
#    )
#plt.legend()
#plt.grid()
#plt.show()