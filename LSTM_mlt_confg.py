import numpy as np
import pandas as pd
import json

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.models import load_model
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import LSTM
#from tensorflow.keras.callbacks import TensorBoard

import settings
import ml_tools
import graphs

ml_tools.clean_output_folders()

with open(settings.conf_m_path) as config_file:
    conf = json.load(config_file)

features = conf["features"]
data = pd.read_excel(settings.ex_data, sheet_name='data')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)
data = data.astype(float)

data_f = data[conf["features"]]
data_y = data[conf["y_var"]]

data_f = data_f.values
data_y = data_y.values
#data.plot(subplots = True)
#data_u = np.concatenate((data[out_var].values,np.flipud(data[out_var].values)))#duplicado y flipiado
dataset = data.values


train_split= ml_tools.data_split(data_f, conf["split_p"])
#Normalize Just Train
#dataset, data_mean, data_std = ml_tools.normaize(dataset)
data_f, data_mean_f, data_std_f = ml_tools.normaize(data_f)
data_y, data_mean_y, data_std_y = ml_tools.normaize(data_y)


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

##############################################################################################
#model = Sequential()
#model.add(LSTM(conf["lstm1"], return_sequences=True, input_shape=x_train.shape[-2:]))
#model.add(LSTM(conf["lstm2"], activation ='relu'))
#model.add(Dense(1))
#print(model.summary())
#model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae',metrics=[conf["metrics"]])
#m_perf = model.fit(x_train, y_train, batch_size = conf["batch_size"], epochs= conf["epoch"], shuffle = False, validation_data = (x_val, y_val))
#
#model.save(settings.m_path+'lstm_m.h5')
#
#graphs.plot_model_metric(m_perf, 'loss', save = True)
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
#it = 16
#graphs.multi_step_plot(x_val[it], y_val[it], yhat[it], STEP, save = True)
#
#graphs.plot_model_learn(data, yhat, save = True)

#ml_tools.save_experiment(True)
################################################################################################

model = Sequential()
model.add(LSTM(conf["lstm1"], return_sequences=True, input_shape=x_train.shape[-2:]))
model.add(LSTM(conf["lstm2"], activation ='relu'))
model.add(Dense(conf["future_target"]))
print(model.summary())
model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae',metrics=[conf["metrics"]])
m_perf = model.fit(x_train, y_train, batch_size = conf["batch_size"], epochs= conf["epoch"], shuffle = False, validation_data = (x_val, y_val))

model.save(settings.m_path+'lstm_m.h5')

graphs.plot_model_metric(m_perf, 'loss', save = True)


for i in conf["metrics"]:
    graphs.plot_model_metric(m_perf, i, save = True)

#model = load_model(settings.m_path+'lstm_m.h5')

yhat = model.predict(x_val)

x_val = x_val[:,:,0]

x_val = ml_tools.desnormalize(x_val, data_mean_y, data_std_y)
y_val = ml_tools.desnormalize(y_val, data_mean_y, data_std_y)

yhat = ml_tools.desnormalize(yhat, data_mean_y, data_std_y)
yhat = ml_tools.model_out_tunep(yhat)

it = 100
graphs.multi_step_plot(x_val[it], y_val[it], yhat[it], STEP, save=True)

yhat_p = yhat[:,0].reshape(len(yhat),1)
graphs.plot_model_learn(data, yhat_p, save = True)

ml_tools.save_experiment(True)

