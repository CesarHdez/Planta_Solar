import numpy as np
import pandas as pd
import json
import datetime
#from keras.models import Sequential
#from keras.models import load_model
#from keras.layers import Dense
#from keras.layers import LSTM
#import tensorflow as tf
#
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.models import load_model
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import LSTM
#from tensorflow.keras.layers import Dropout
#
#from tensorflow.keras.regularizers import Regularizer
#
#from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.callbacks import ReduceLROnPlateau
#from tensorflow.keras.callbacks import TensorBoard

import model_mk
import settings
import ml_tools
import graphs
#import u_models

ml_tools.clean_output_folders()

with open(settings.conf_m_path) as config_file:
    conf = json.load(config_file)

features = conf["features"]
data = pd.read_excel(settings.ex_data, sheet_name='data')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)
data = data[:-140]
data = data.astype(float)

data_f = data[conf["features"]]
data_y = data[conf["y_var"]]

data_f = data_f.values
data_y = data_y.values
#data.plot(subplots = True)
#data_u = np.concatenate((data[out_var].values,np.flipud(data[out_var].values)))#duplicado y flipiado



train_split= ml_tools.data_split(data_f, conf["split_p"])

data_f, data_mean_f, data_std_f = ml_tools.normaize(data_f)
data_y, data_mean_y, data_std_y = ml_tools.normaize(data_y)

##############################################################

past_hist= conf["past_hist"]
future_target = conf["future_target"]
STEP = conf["step"]

out_var_num = data.columns.get_loc(conf["y_var"])

x_train, y_train = ml_tools.multivariate_data(data_f, data_y, 0,
                                                 train_split, past_hist,
                                                 future_target, STEP)
x_val, y_val = ml_tools.multivariate_data(data_f, data_y,
                                             train_split, None, past_hist,
                                             future_target, STEP)

#print ('Single window of past history : {}'.format(x_train[0].shape))
#print ('\n Target temperature to predict : {}'.format(y_train[0].shape))

#graphs.multi_step_plot(x_train[0], y_train[0], np.array([0]), STEP)


model, m_perf = model_mk.model_maker(conf, x_train, y_train, x_val, y_val)

model.save(settings.m_path+conf['type']+'_m'+'.h5')

graphs.plot_model_metric(m_perf, 'loss', save = True)

for i in conf["metrics"]:
    graphs.plot_model_metric(m_perf, i, save = True)

#model = load_model(settings.m_path+'lstm_m.h5')

yhat = model.predict(x_val)

x_val = x_val[:,:,0]

x_val = ml_tools.desnormalize(x_val, data_mean_y, data_std_y)
y_val = ml_tools.desnormalize(y_val, data_mean_y, data_std_y)

yhat = ml_tools.desnormalize(yhat, data_mean_y, data_std_y)
#yhat = ml_tools.model_out_tunep(yhat)

#it = 16
#graphs.multi_step_plot(x_val[it], y_val[it], yhat[it], STEP, save = True)

graphs.plot_model_learn(data, yhat, save = True)

yhat = ml_tools.model_out_tunep(yhat)
graphs.plot_model_learn(data, yhat, save = True)

graphs.plot_scatter_learn(data, yhat, save = True)

relat = ml_tools.forecast_relation(data, yhat)
print(relat[:30])
cor = relat.astype(float).corr(method = 'pearson')
print('Correlation: ', cor)


ml_tools.save_experiment(True)

########################################################
#data_2_append = u_models.data_2_append[conf["features"]]
#data_f = data[conf["features"]]
#new_data_f = pd.concat([data_f, data_2_append])
#data_f = new_data_f.values
#data_f, data_mean_f, data_std_f = ml_tools.normaize(data_f)
#start_index = len(data_f)-len(data_2_append) - past_hist
#x_new = ml_tools.multivariate_data_2(data_f, start_index, len(data_f), past_hist, future_target, STEP)
#
#yhat = model.predict(x_new)
#
#yhat = ml_tools.desnormalize(yhat, data_mean_y, data_std_y)
#
#yhat = ml_tools.model_out_tunep(yhat)
#graphs.plot_model_learn(data, yhat, save = True)
#
#yhat=yhat.reshape(24,)
#graphs.plot_next_forecast(data, yhat, conf["n_ahead"], save=True)
#
#ml_tools.save_experiment(True)

###################################################################

#conf["type"]= "Custom2"
#conf["loss"]= "mae"
#conf["future_target"]= 25
#
#past_hist= conf["past_hist"]
#future_target = conf["future_target"]
#STEP = conf["step"]
#
#x_train, y_train = ml_tools.multivariate_data(data_f, data_y, 0,
#                                                 train_split, past_hist,
#                                                 future_target, STEP)
#x_val, y_val = ml_tools.multivariate_data(data_f, data_y,
#                                             train_split, None, past_hist,
#                                             future_target, STEP)
#
#model, m_perf = model_mk.model_maker(conf, x_train, y_train, x_val, y_val)
#
#model.save(settings.m_path+'lstm_m.h5')
#
#graphs.plot_model_metric(m_perf, 'loss', save = True)
#
#
#for i in conf["metrics"]:
#    graphs.plot_model_metric(m_perf, i, save = True)
#
##model = load_model(settings.m_path+'lstm_m.h5')
#
#yhat = model.predict(x_val)
#
#x_val = x_val[:,:,0]
#
#x_val = ml_tools.desnormalize(x_val, data_mean_y, data_std_y)
#y_val = ml_tools.desnormalize(y_val, data_mean_y, data_std_y)
#
#yhat = ml_tools.desnormalize(yhat, data_mean_y, data_std_y)
#yhat = ml_tools.model_out_tunep(yhat)
#
#it = 100
#graphs.multi_step_plot(x_val[it], y_val[it], yhat[it], STEP, save=True)
#
#yhat_p = yhat[:,0].reshape(len(yhat),1)
#graphs.plot_model_learn(data, yhat_p, save = True)
#
#ml_tools.save_experiment(True)
#
