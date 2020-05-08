#import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import math
#import keras

import settings
import tools
import ml_tools


#Obtener los nombres de los archivos para procesar
files_name = tools.ls2(settings.path)
#crear dataframe con los headers de acuerdo a la data
full_data = pd.DataFrame(columns= settings.headers_list)
#agregar al dataframe cada mes(archivo de excel)
for i in files_name:
    full_data =tools.add_data_month(full_data, settings.path +'/' + i)

#Dar formato al dataframe: setear fecha como indice y ordenar los datos cronológicamentes
full_data = tools.format_dataframe(full_data, settings.headers_list[0])

#cambiar valores de radiación solar negativos a zero
full_data = tools.neg_irrad_2_zero(full_data)

#cambiar los datos negativos de generación de energía a positivos
full_data = tools.negative_to_positive(full_data, 'ENERGY')

#obtener solo las horas de sol
full_data = tools.full_data_sun_hours(full_data, 'ENERGY')

#Eliminación de columnas con las que no se va a trabajar
full_data = tools.delete_cols(full_data, settings.cols2delete)

#full_data contiene los datos con frecuencia de una hora.
#full_data en distintas frecuencias
daily = full_data.resample('D').mean()
weekly = full_data.resample('W').mean()
monthly = full_data.resample('M').mean()



#Gráficas
#---------
#sbn.set(rc={'figure.figsize':(15, 5)})
#daily['ENERGY'].plot(linewidth=1)

#Graficarlas todas
#fig,eje= plt.subplots()
#for i in ['WS1','IRRAD1','TEMP1','WANG', 'ENERGY']:
#    eje.plot(daily[i],label=i)
#    eje.set_ylim(0,7000)
#    eje.legend()
#    eje.set_ylabel('Producción (GWh)')
#    eje.set_title('Tendencias en la Producción de electricidad')

#fig,eje = plt.subplots()
#eje.plot(daily['ENERGY'],color='black',label='Consumo')
#daily[['IRRAD1','WS1']].plot.area(ax=eje,linewidth=0)
#eje.legend()
#eje.set_ylabel('Total Mensual (GWh)')


#values = dataset.values
## specify columns to plot
#groups = [0, 1, 2, 3, 5, 6, 7]
#i = 1
## plot each column
#pyplot.figure()
#for group in groups:
#	pyplot.subplot(len(groups), 1, i)
#	pyplot.plot(values[:, group])
#	pyplot.title(dataset.columns[group], y=0.5, loc='right')
#	i += 1
#pyplot.show()


fig,eje = plt.subplots()
eje.plot(daily['ENERGY'],color='black',label='Energy')
eje2 = eje.twinx()
eje2.plot(daily['IRRAD1'],color='red',label='Solar Rad')
eje.legend()
eje.set_ylabel('Relacion Radiacion- Energía')

f, ax = plt.subplots(figsize=(10, 8))
cor = daily.corr()
sbn.heatmap(cor, mask=np.zeros_like(cor, dtype=np.bool), cmap=sbn.diverging_palette(220, 10, as_cmap=True),
               square=True, ax=ax)

#--------------------------------------------------------
#A partir de  aqui minería de datos
#--------------------------------------------------------

#Settings
lag_size = 6
time_steps = 3
train_test_split_percent = 70
n_epochs = 10
batch_size = 31
n_neurons = 3

energy = daily['ENERGY'].values

raw_data = daily.values
n_features = raw_data.shape[1]
n_features = n_features * lag_size

diff_values = ml_tools.difference(raw_data, 1)
diff_energy = ml_tools.difference(energy, 1)

supervised_df, supervised_arr = ml_tools.timeseries_to_supervised(diff_values, diff_energy, lag_size, time_steps)

scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(supervised_arr)

train, test, split_limit = ml_tools.data_split(supervised_arr, train_test_split_percent)

model = ml_tools.fit_mlp(train, batch_size, n_epochs, n_neurons, time_steps,
                lag_size, n_features)

X_train = pd.DataFrame(data=train[:, 0:n_features]).values

test_X, test_y = test[:, 0:n_features], test[:, n_features:]

yhat = np.array(model.predict(test_X))

yhat = ml_tools.inverse_transform(energy, test_X, yhat, n_features, scaler)
test_y = ml_tools.inverse_transform(energy, test_X, test_y, n_features, scaler)

for i in range(time_steps):
    actual = test_y[:, i]
    predicted = yhat[:, i]
    rmse = math.sqrt(metrics.mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    print("-----------------------------")
    print('t+%d RMSE: %f' % ((i+1), rmse))
    print('t+%d mape: %f' % ((i+1), mape))

print("-----------------------------")
# line plot of observed vs predicted
plt.figure(1)
split_limit = split_limit+time_steps+1
plt.plot(energy[split_limit:-time_steps], label = "Real solar Radiation")
plt.plot(yhat[:,-1], label = "Predicted solar Radiation")
plt.legend()
plt.show()

plt.figure(1)
plt.scatter(test_y[:,-1],yhat[:,-1],c='r', alpha=0.5, label='Solar Radiation')
plt.xlabel("Real target values")
plt.ylabel("Predicted target values")
axes = plt.gca()
m, b = np.polyfit(test_y[:,-1], yhat[:,-1], 1)
x_plot = np.linspace(axes.get_xlim()[0], axes.get_xlim()[1], 100)
plt.plot(x_plot, m*x_plot + b,'-')
plt.legend(loc='upper left')
plt.show()