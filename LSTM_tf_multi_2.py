import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step = False):
    data = []
    labels = []
    
    start_index = start_index + history_size
    if end_index is None or end_index > (len(dataset) - target_size):
        end_index = len(dataset) - target_size
        
    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])
        
        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])
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

def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 1]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()

def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()

def desnormalize(data, mean, std):
    des_norm = (data*std)+mean
    return des_norm



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


train_split= data_split(dataset, 90)
#Normalize Just Train
data_mean = dataset[:train_split].mean(axis =0)
data_std = dataset[:train_split].std(axis=0)
dataset = (dataset - data_mean)/data_std


past_hist= 120
future_target = 24
STEP = 6


x_train, y_train = multivariate_data(dataset, dataset[:, 1], 0,
                                                 train_split, past_hist,
                                                 future_target, STEP)
x_val, y_val = multivariate_data(dataset, dataset[:, 1],
                                             train_split, None, past_hist,
                                             future_target, STEP)

print ('Single window of past history : {}'.format(x_train[0].shape))
print ('\n Target temperature to predict : {}'.format(y_train[0].shape))

multi_step_plot(x_train[0], y_train[0], np.array([0]))


model = Sequential()
model.add(LSTM(120, return_sequences=True, input_shape=x_train.shape[-2:]))
model.add(LSTM(16, activation ='relu'))
model.add(Dense(24))
model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
m_performance = model.fit(x_train, y_train, batch_size = 256, epochs= 10, shuffle = False, validation_data = (x_val, y_val))

plot_train_history(m_performance, 'Single Step Training and validation loss')

yhat = model.predict(x_val)

it = 0
multi_step_plot(x_val[it], y_val[it], yhat[it])

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