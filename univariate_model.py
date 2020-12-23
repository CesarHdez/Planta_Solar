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

data = pd.read_excel(settings.ex_data, sheet_name='data')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)
#data = data[:-120]
data = data.astype(float)
#data = data['ENERGY']
data[conf["y_var"]].astype(float)
#------------------------------------------------------------
#data.plot(subplots = True)
#data_u = np.concatenate((data[out_var].values,np.flipud(data[out_var].values)))#duplicado y flipiado
data_u = data[conf["y_var"]].values



data_u, data_mean, data_std = ml_tools.normalize(data_u)

train_split= ml_tools.data_split(data_u, conf["split_p"])

u_past_hist= conf["past_hist"]
u_future_traget = conf["future_target"]

x_train, y_train = ml_tools.univariate_data(data_u, 0, train_split, u_past_hist, u_future_traget)
x_val, y_val = ml_tools.univariate_data(data_u, train_split, None, u_past_hist, u_future_traget)

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
#graphs.show_plot([x_val[it], y_val[it], yhat[it]], 0,conf["type"], True)

yhat = ml_tools.desnormalize(yhat, data_mean, data_std)

yhat = ml_tools.model_out_tunep(yhat)

graphs.plot_model_learn(data, yhat, conf["y_var"], True)

graphs.plot_model_learn(data[500:-137], yhat[500:-137], conf["y_var"], True)

graphs.plot_scatter_learn(data, yhat, save = True)

relat = ml_tools.forecast_relation(data, yhat)
ml_tools.save_perf(settings.m_path+conf['type']+'_u'+'_fc_dt'+'.pkl', relat)
print(relat[:30])
cor = relat.astype(float).corr(method = 'pearson')
print('Correlation: ', cor)
rr = ml_tools.det_coef(relat["ENERGY"].values, relat["forecast"].values)
print("R2 coef: ",rr)


y_true = np.array(relat["ENERGY"])
y_pred = np.array(relat["forecast"])
MSE = np.square(np.subtract(y_true, y_pred)).mean()
RMSE = sqrt(MSE)
print("MSE: ", MSE)
print("RMSE: ", RMSE)

#print(ml_tools.get_model_stats(m_perf.history))
#----------------------------------------------------
metrics = ["mse", "mae", "mape", "root_mean_squared_error"]
metric_list=[]
for k in metrics: 
    metric_list.append(m_perf.history[k][-1])

#columns_l=['name','corr', 'det'] + metrics
#models_stats= pd.DataFrame(columns = columns_l)
models_stats= pd.DataFrame()
models_stats['name']=[conf["optimizer"]]
models_stats['corr']=[cor.iloc[0][1]]
models_stats['det']=[rr]

for i in range(len(metrics)):
    if metrics[i] == "root_mean_squared_error":
        models_stats['rmse']=metric_list[i]
    else:
        models_stats[metrics[i]]=metric_list[i]
models_stats.to_excel(settings.m_path+conf['type']+'models_stats.xlsx', sheet_name='stats')
print(models_stats)  

ml_tools.save_experiment(conf)
#---------------------------------------
print("Análisis diario")
daily = relat.resample('D').sum()[1:-1]
graphs.plot_model_learn_days(daily, True)
graphs.plot_scatter_learn_days(daily, daily['forecast'].values, save = True)
graphs.plot_scatter_learn_days_2(daily)
ml_tools.save_perf(settings.m_path+conf['type']+'_u'+'_fc_dt_d'+'.pkl', daily)
print(daily[:30])
cor_d = daily.astype(float).corr(method = 'pearson')
print('Correlation: ', cor_d)
rr_d = ml_tools.det_coef(daily["ENERGY"].values, daily["forecast"].values)
print("R2 coef: ",rr_d)
#------------------------------------------

####################################
##Predecir los N siguientes valores

#u_past_hist= conf["past_hist"]
#u_future_traget = conf["future_target"]
#
#train_split= ml_tools.data_split(data_u, 100)
#
#x_train, y_train = ml_tools.univariate_data(data_u, 0, train_split, u_past_hist, u_future_traget)
#
#model, m_perf = model_mk.model_maker(conf, x_train, y_train)
#
#graphs.plot_model_metric(m_perf, 'loss', save = True)
#
#n_ahead = conf["n_ahead"]
#last_input= x_train[-1]
#
#yhat = ml_tools.predict_n_ahead(model, n_ahead, last_input)
#
#yhat = ml_tools.desnormalize(np.array(yhat), data_mean, data_std)
#
#yhat = ml_tools.model_out_tunep(yhat)
#
#graphs.plot_next_forecast(data, yhat, n_ahead, save=True)
#
#fc = ml_tools.forecast_dataframe(data, yhat, n_ahead)
#fc =fc.iloc[-49:]
#print(fc)

#ml_tools.save_experiment()


####################################
#u_past_hist= conf["past_hist"]
#u_future_traget = conf["future_target"]
#
#train_split= ml_tools.data_split(data_u, 100)
#
#x_train, y_train = ml_tools.univariate_data(data_u, 0, train_split, u_past_hist, u_future_traget)
#
#model, m_perf = model_mk.model_maker(conf, x_train, y_train)
#
#graphs.plot_model_metric(m_perf, 'loss', save = True)
#
#yhat= model.predict(x_train)
#
#yhat = ml_tools.desnormalize(yhat, data_mean, data_std)
#
#yhat = ml_tools.model_out_tunep(yhat)
#
#graphs.plot_model_learn(data, yhat, conf["y_var"], True)
#
#relat = ml_tools.forecast_relation(data, yhat)
#print(relat[:30])
#cor = relat.astype(float).corr(method = 'pearson')
#print('Correlation: ', cor)
#rr = ml_tools.det_coef(relat["ENERGY"].values, relat["forecast"].values)
#print("R2 coef: ",rr)
#
#y_true = np.array(relat["ENERGY"])
#y_pred = np.array(relat["forecast"])
#MSE = np.square(np.subtract(y_true, y_pred)).mean()
#RMSE = sqrt(MSE)
#print("MSE: ", MSE)
#print("RMSE: ", RMSE)
#
#print(ml_tools.get_model_stats(m_perf.history))




#n_ahead = conf["n_ahead"]
#last_input= x_train[-1]
#
#yhat = ml_tools.predict_n_ahead(model, n_ahead, last_input)
#
#yhat = ml_tools.desnormalize(np.array(yhat), data_mean, data_std)
#
#yhat = ml_tools.model_out_tunep(yhat)
#
#graphs.plot_next_forecast(data, yhat, n_ahead, save=True)
#
#fc = ml_tools.forecast_dataframe(data, yhat, n_ahead)
#fc =fc.iloc[-49:]
#print(fc)
#
#ml_tools.save_experiment()