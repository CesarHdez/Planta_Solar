import numpy as np
import pandas as pd
import json
import datetime

from math import sqrt

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.models import load_model
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import LSTM
#from tensorflow.keras.layers import SimpleRNN
#from tensorflow.keras.layers import GRU
#from tensorflow.keras.layers import Dropout
#
#
#from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.callbacks import ReduceLROnPlateau
#from tensorflow.keras.callbacks import TensorBoard
#import tensorflow as tf

#from keras.models import Sequential
#from keras.models import load_model
#from keras.layers import Dense
#from keras.layers import LSTM
import model_mk
import settings
import graphs
import ml_tools
import tools

ml_tools.clean_output_folders()

with open(settings.conf_u_path) as config_file:
    conf = json.load(config_file)

data = pd.read_excel(settings.ex_data_2, sheet_name='data')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)
#data = data[:-120]
data = data.astype(float)
#data = data['ENERGY']
data[conf["y_var"]].astype(float)
#data.plot(subplots = True)
#data_u = np.concatenate((data[out_var].values,np.flipud(data[out_var].values)))#duplicado y flipiado
data_y = data[conf["y_var"]].values
data_x = data["IRRAD1"].values



data_x, data_mean, data_std = ml_tools.normalize(data_x)
data_y, data_mean, data_std = ml_tools.normalize(data_y)

train_split= ml_tools.data_split(data_x, conf["split_p"])

u_past_hist= conf["past_hist"]
u_future_traget = conf["future_target"]

x_train, y_train = ml_tools.univariate_data_2(data_x, data_y, 0, train_split, u_past_hist, u_future_traget)
x_val, y_val = ml_tools.univariate_data_2(data_x, data_y, train_split, None, u_past_hist, u_future_traget)




model, m_perf = model_mk.model_maker(conf, x_train, y_train, x_val, y_val)

model.save(settings.m_path+conf['type']+'_u'+'.h5')

ml_tools.save_perf(settings.m_path+conf['type']+'_u'+'.pkl', m_perf.history)

#mpkl = ml_tools.load_perf(settings.m_path+conf['type']+'_u'+'.pkl')

graphs.plot_model_metric(m_perf, 'loss', save = True)

graphs.plot_model_metric(m_perf, 'root_mean_squared_error', save = True)

for i in conf["metrics"]:
    graphs.plot_model_metric(m_perf, i, save = True)

#model = load_model(settings.m_path+'lstm_u.h5')

yhat= model.predict(x_val)
#yhat = [y[0] for y in model.predict(x_val)]

it = 17
graphs.show_plot([x_val[it], y_val[it], yhat[it]], 0,conf["type"], True)

yhat = ml_tools.desnormalize(yhat, data_mean, data_std)

yhat = ml_tools.model_out_tunep(yhat)

graphs.plot_model_learn(data, yhat, conf["y_var"], True)

graphs.plot_model_learn(data[500:-137], yhat[500:-137], conf["y_var"], True)

graphs.plot_scatter_learn(data, yhat, save = True)

relat = ml_tools.forecast_relation(data, yhat)
print(relat[:30])
cor = relat.astype(float).corr(method = 'pearson')
print('Correlation: ', cor)

y_true = np.array(relat["ENERGY"])
y_pred = np.array(relat["forecast"])
MSE = np.square(np.subtract(y_true, y_pred)).mean()
RMSE = sqrt(MSE)
print("MSE: ", MSE)
print("RMSE: ", RMSE)

print(ml_tools.get_model_stats(m_perf.history))

ml_tools.save_experiment(conf)
