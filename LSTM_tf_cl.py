import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd
import json
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM


import settings
import tools
import graphs
import ml_tools

#with open('config.json') as config_file:
#    conf = json.load(config_file)
#
#print(conf["y_var"])

data = pd.read_excel('/content/solar_plant/full_data.xlsx', sheet_name='data')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)
data = data.astype(float)
#data = data['ENERGY']
out_var = 'ENERGY'
data[out_var].astype(float)
#data.plot(subplots = True)
#data_u = np.concatenate((data[out_var].values,np.flipud(data[out_var].values)))#duplicado y flipiado
data_u = data[out_var].values


data_u, data_mean, data_std = ml_tools.normaize(data_u)

train_split= ml_tools.data_split(data_u, 90)

u_past_hist= 120
u_future_traget = 0

x_train, y_train = ml_tools.univariate_data(data_u, 0, train_split, u_past_hist, u_future_traget)
x_val, y_val = ml_tools.univariate_data(data_u, train_split, None, u_past_hist, u_future_traget)

model = Sequential()
model.add(LSTM(120, input_shape=x_train.shape[-2:]))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mae', metrics=['mse', 'mae', 'mape', 'cosine'])
print(model.summary())
m_perf = model.fit(x_train, y_train, batch_size = 256, epochs= 10, shuffle = False, validation_data = (x_val, y_val))
#model.save(settings.m_path+'lstm_u.h5')

graphs.plot_model_metric(m_perf, 'loss', False)

#model = load_model('./models/lstm_u.h5')

yhat= model.predict(x_val)
#yhat = [y[0] for y in model.predict(x_val)]

it = 17
graphs.show_plot([x_val[it], y_val[it], yhat[it]], 0,'LSTM model')

yhat = ml_tools.desnormalize(yhat, data_mean, data_std)

graphs.plot_model_learn(data, yhat)

####################################
# Predecir los N siguientes valores


u_past_hist= 120
u_future_traget = 0
train_split= ml_tools.data_split(data_u, 100)
x_train, y_train = ml_tools.univariate_data(data_u, 0, train_split, u_past_hist, u_future_traget)

model = Sequential()
model.add(LSTM(120, input_shape=x_train.shape[-2:]))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mae')
m_perf = model.fit(x_train, y_train, batch_size = 256, epochs= 10, shuffle = False)

graphs.plot_model_metric(m_perf, 'loss', save = False)

n_ahead = 24
last_input= x_train[-1]

yhat = ml_tools.predict_n_ahead(model, n_ahead, last_input)

yhat = ml_tools.desnormalize(np.array(yhat), data_mean, data_std)

yhat = ml_tools.model_out_tunep(yhat)

graphs.plot_next_forecast(data, yhat, n_ahead)

fc = ml_tools.forecast_dataframe(data, yhat, n_ahead)
fc =fc.iloc[-49:]
print(fc)

