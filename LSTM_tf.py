import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None or end_index > (len(dataset) - target_size):
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        #Reshape data to (history, 1)
        data. append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)

def data_split(array, percent):
	limit = int(len(array) * percent / 100)
	return limit

def create_time_steps(length):
  return list(range(-length, 0))

def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt

def desnormalize(data, mean, std):
    des_norm = (data*std)+mean
    return des_norm

data = pd.read_excel('full_data.xlsx', sheet_name='data')
data['DateTime'] = pd.to_datetime(data['DateTime'])
#data.set_index('DateTime', inplace=True)
#data = data.astype(float)
#data = data['ENERGY']
out_var = 'ENERGY'
data[out_var].astype(float)
#data.plot(subplots = True)
#data_u = np.concatenate((data[out_var].values,np.flipud(data[out_var].values)))#duplicado y flipiado
data_u = data[out_var].values


train_split= data_split(data_u, 90)
#Normalize Just Train
data_train_mean = data_u[:train_split].mean()
data_train_std = data_u[:train_split].std()
data_u = (data_u - data_train_mean)/data_train_std


u_past_hist= 24
u_future_traget = 0

x_train, y_train = univariate_data(data_u, 0, train_split, u_past_hist, u_future_traget)
x_val, y_val = univariate_data(data_u, train_split, None, u_past_hist, u_future_traget)

print ('Single window of past history')
print (x_train[0])
print ('\n Target temperature to predict')
print (y_train[0])

it = 11
show_plot([x_train[it], y_train[it]], 0, 'Sample Example')

model = Sequential()
model.add(LSTM(120, input_shape=x_train.shape[-2:]))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mae')
model.fit(x_train, y_train, batch_size = 256, epochs= 10, shuffle = False, validation_data = (x_val, y_val))


yhat = model.predict(x_val)

it = 0
plot = show_plot([x_val[it], y_val[it], yhat[it]], 0,'Simple LSTM model')
plot.show()

#Constructing the forecast dataframe
fc = data.tail(len(yhat)).copy()
#fc.reset_index(inplace=True)

yhat = desnormalize(yhat, data_train_mean, data_train_std)

fc['forecast'] = yhat
fc = fc[['DateTime', 'ENERGY', 'forecast']]
# Ploting the forecasts
plt.figure(figsize=(20, 8))
for dtype in ['ENERGY', 'forecast']:
    plt.plot(
        'DateTime',
        dtype,
        data=fc,
        label=dtype,
        alpha=0.8
    )
plt.legend()
plt.grid()
plt.show()  





