#import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn import metrics
import math
#import keras

import settings
import tools
import ml_tools


#Obtener los nombres de los archivos para procesar
files_name = tools.ls2(settings.path)
#crear dataframe con los headers de acuerdo a la data
full_ = pd.DataFrame(columns= settings.headers_list)
#agregar al dataframe cada mes(archivo de excel)
for i in files_name:
    full_ =tools.add_data_month(full_, settings.path +'/' + i)

#Dar formato al dataframe: setear fecha como indice y ordenar los datos cronológicamentes
full_data = tools.format_dataframe(full_, settings.headers_list[0])

#cambiar valores de radiación solar negativos a zero
full_data = tools.neg_irrad_2_zero(full_data)

#cambiar los datos negativos de generación de energía a positivos
full_data = tools.negative_to_positive(full_data, 'ENERGY')

#cambiar valores extremos
full_data = tools.change_outliers_values(full_data, 'ENERGY')

#obtener solo las horas de sol
full_data = tools.full_data_sun_hours(full_data, 'ENERGY')

#Eliminación de columnas con las que no se va a trabajar
full_data = tools.delete_cols(full_data, settings.cols2delete)

#full_data contiene los datos con frecuencia de una hora.
#full_data en distintas frecuencias
daily = full_data.resample('D').mean()
weekly = full_data.resample('W').mean()
monthly = full_data.resample('M').mean()

dataset = daily #dataset con el que se va a trabaja

#------------------------------------------------
#Gráficos
#------------------------------------------------


#grafico para series temporales
#sbn.set(rc={'figure.figsize':(15, 5)})
#dataset['ENERGY'].plot(linewidth=1)

#Graficarlas todas en el mismo graph no buena
#fig,eje= plt.subplots()
#for i in ['WS1','IRRAD1','TEMP1','WANG', 'ENERGY']:
#    eje.plot(dataset[i],label=i)
#    eje.set_ylim(0,7000)
#    eje.legend()
#    eje.set_ylabel('Producción (GWh)')
#    eje.set_title('Tendencias en la Producción de electricidad')

#grafica varias juntas con el modo de una de area.
#fig,eje = plt.subplots()
#eje.plot(dataset['ENERGY'],color='black',label='Consumo')
#dataset[['IRRAD1','WS1']].plot.area(ax=eje,linewidth=0)
#eje.legend()
#eje.set_ylabel('Total Mensual (GWh)')

#graficar todas por separado

values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3, 4]
i = 1
# plot each column
plt.figure()
for group in groups:
	plt.subplot(len(groups), 1, i)
	plt.plot(values[:, group])
	plt.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
plt.show()


# fig,eje = plt.subplots()
# eje.plot(dataset['ENERGY'],color='black',label='Energy')
# eje2 = eje.twinx()
# eje2.plot(dataset['IRRAD1'],color='red',label='Solar Rad')
# eje.legend()
# eje.set_ylabel('Relacion Radiacion- Energía')

# f, ax = plt.subplots(figsize=(10, 8))
cor = dataset.corr()
print(cor)
# sbn.heatmap(cor, mask=np.zeros_like(cor, dtype=np.bool), cmap=sbn.diverging_palette(220, 10, as_cmap=True),
#                square=True, ax=ax)

#--------------------------------------------------------
#A partir de  aqui minería de datos
#--------------------------------------------------------
lag_size = 4
train_test_split_percent = 70

values = dataset.values
encoder = LabelEncoder()
values[:,lag_size] = encoder.fit_transform(values[:,lag_size])
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = ml_tools.series_to_supervised(scaled, 1, 1)

reframed.drop(reframed.columns[[1,2,3,4]], axis=1, inplace=True)

values = reframed.values
# split into train and test sets
train, test, split_limit = ml_tools.data_split(values, train_test_split_percent)

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

#Evaluación
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = math.sqrt(metrics.mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)



###########Trabajo de GitHub_1#############
#Settings
#lag_size = 6
#time_steps = 3
#train_test_split_percent = 70
#n_epochs = 10
#batch_size = 31
#n_neurons = 3
#
#energy = dataset['ENERGY'].values
#
#raw_data = dataset.values
#n_features = raw_data.shape[1]
#n_features = n_features * lag_size
#
#diff_values = ml_tools.difference(raw_data, 1)
#diff_energy = ml_tools.difference(energy, 1)
#
#supervised_df, supervised_arr = ml_tools.timeseries_to_supervised_m(diff_values, diff_energy, lag_size, time_steps)
#
#scaler = MinMaxScaler(feature_range=(-1, 1))
#scaled = scaler.fit_transform(supervised_arr)
#
#train, test, split_limit = ml_tools.data_split(supervised_arr, train_test_split_percent)
#
#model = ml_tools.fit_mlp(train, batch_size, n_epochs, n_neurons, time_steps,
#                lag_size, n_features)
#
#X_train = pd.DataFrame(data=train[:, 0:n_features]).values
#
#test_X, test_y = test[:, 0:n_features], test[:, n_features:]
#
#yhat = np.array(model.predict(test_X))
#
#yhat = ml_tools.inverse_transform(energy, test_X, yhat, n_features, scaler)
#test_y = ml_tools.inverse_transform(energy, test_X, test_y, n_features, scaler)
#
#for i in range(time_steps):
#    actual = test_y[:, i]
#    predicted = yhat[:, i]
#    rmse = math.sqrt(metrics.mean_squared_error(actual, predicted))
#    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
#    print("-----------------------------")
#    print('t+%d RMSE: %f' % ((i+1), rmse))
#    print('t+%d mape: %f' % ((i+1), mape))
#
#print("-----------------------------")
## line plot of observed vs predicted
#plt.figure(1)
#split_limit = split_limit+time_steps+1
#plt.plot(energy[split_limit:-time_steps], label = "Real solar Radiation")
#plt.plot(yhat[:,-1], label = "Predicted solar Radiation")
#plt.legend()
#plt.show()
#
#plt.figure(1)
#plt.scatter(test_y[:,-1],yhat[:,-1],c='r', alpha=0.5, label='Solar Radiation')
#plt.xlabel("Real target values")
#plt.ylabel("Predicted target values")
#axes = plt.gca()
#m, b = np.polyfit(test_y[:,-1], yhat[:,-1], 1)
#x_plot = np.linspace(axes.get_xlim()[0], axes.get_xlim()[1], 100)
#plt.plot(x_plot, m*x_plot + b,'-')
#plt.legend(loc='upper left')
#plt.show()