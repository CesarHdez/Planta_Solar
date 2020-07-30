import numpy as np
import pandas as pd
import json
import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten


from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf

#from keras.models import Sequential
#from keras.models import load_model
#from keras.layers import Dense
#from keras.layers import LSTM

import settings
import graphs
import ml_tools

ml_tools.clean_output_folders()

with open(settings.conf_path_m_mpl) as config_file:
    conf = json.load(config_file)

data = pd.read_excel(settings.ex_data, sheet_name='data')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)
data = data.astype(float)
#data = data['ENERGY']
###################################

data_f = data[conf["features"]]
data_y = data[conf["y_var"]]

data_f = data_f.values
data_y = data_y.values
#data.plot(subplots = True)
#data_u = np.concatenate((data[out_var].values,np.flipud(data[out_var].values)))#duplicado y flipiado



train_split= ml_tools.data_split(data_f, conf["split_p"])

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

##################################
#data[conf["y_var"]].astype(float)
##data.plot(subplots = True)
##data_u = np.concatenate((data[out_var].values,np.flipud(data[out_var].values)))#duplicado y flipiado
#data_u = data[conf["y_var"]].values
#
#
#data_u, data_mean, data_std = ml_tools.normaize(data_u)
#
#train_split= ml_tools.data_split(data_u, conf["split_p"])
#
#u_past_hist= conf["past_hist"]
#u_future_traget = conf["future_target"]
#
#x_train, y_train = ml_tools.univariate_data(data_u, 0, train_split, u_past_hist, u_future_traget)
#x_val, y_val = ml_tools.univariate_data(data_u, train_split, None, u_past_hist, u_future_traget)


model = Sequential()
if conf["layer2"] == 0:
    model.add(Dense(conf["layer1"], activation =conf["act_func"], input_shape=x_train.shape[-2:]))
    model.add(Dropout(conf["dropout"]))
elif conf["layer3"] == 0:
    model.add(Dense(conf["layer1"], activation =conf["act_func"], input_shape=x_train.shape[-2:]))
    #kernel_regularizer=tf.keras.regularizers.l2(conf["l2_reg"])
    model.add(Dropout(conf["dropout"]))
    model.add(Dense(conf["layer2"], activation =conf["act_func"]))
    model.add(Dropout(conf["dropout"]))
else:
    model.add(Dense(conf["layer1"], activation =conf["act_func"], input_shape=x_train.shape[-2:]))
    model.add(Dropout(conf["dropout"]))
    model.add(Dense(conf["layer2"], activation =conf["act_func"]))
    model.add(Dropout(conf["dropout"]))
    model.add(Dense(conf["layer3"], activation =conf["act_func"]))
    model.add(Dropout(conf["dropout"]))
model.add(Flatten())
model.add(Dense(1))
print(model.summary())
if conf["callbacks"] == 1:
    print("using callbacks...") #tensorboard --logdir=logs/fit
    log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_s = EarlyStopping('loss', patience = conf["early_s"], mode = 'min')
    lr_red = ReduceLROnPlateau('loss', patince= conf["early_s"], mode = 'min', verbose= conf["early_s"])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=conf["lr"]), loss=conf["loss"],metrics=[conf["metrics"]])
    m_perf = model.fit(x_train, y_train, batch_size = conf["batch_size"], epochs= conf["epoch"], shuffle = False, validation_data = (x_val, y_val), callbacks=[early_s, lr_red, tensorboard])
else:
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=conf["loss"],metrics=[conf["metrics"]])
    m_perf = model.fit(x_train, y_train, batch_size = conf["batch_size"], epochs= conf["epoch"], shuffle = False, validation_data = (x_val, y_val))

model.save(settings.m_path+'mpl_u.h5')

graphs.plot_model_metric(m_perf, 'loss', save = True)

for i in conf["metrics"]:
    graphs.plot_model_metric(m_perf, i, save = True)

#model = load_model(settings.m_path+'lstm_u.h5')

yhat= model.predict(x_val)
#yhat = [y[0] for y in model.predict(x_val)]

#it = 17
#graphs.show_plot([x_val[it], y_val[it], yhat[it]], 0,'LSTM model', True)
##############
x_val = x_val[:,:,0]

x_val = ml_tools.desnormalize(x_val, data_mean_y, data_std_y)
y_val = ml_tools.desnormalize(y_val, data_mean_y, data_std_y)

yhat = ml_tools.desnormalize(yhat, data_mean_y, data_std_y)

yhat = ml_tools.model_out_tunep(yhat)

graphs.plot_model_learn(data, yhat, save = True)

graphs.plot_scatter_learn(data, yhat, save = True)

graphs.plot_scatter_learn(data, yhat, save = True)

relat = ml_tools.forecast_relation(data, yhat)
print(relat[:30])
cor = relat.astype(float).corr(method = 'pearson')
print('Correlation: ', cor)

ml_tools.save_experiment()
##############
#yhat = ml_tools.desnormalize(yhat, data_mean, data_std)
#
#yhat = ml_tools.model_out_tunep(yhat)
#
#graphs.plot_model_learn(data, yhat, True)
#
#graphs.plot_scatter_learn(data, yhat, save = True)
#
#relat = ml_tools.forecast_relation(data, yhat)
#print(relat[:30])
#cor = relat.astype(float).corr(method = 'pearson')
#print('Correlation: ', cor)
#
#ml_tools.save_experiment()

####################################
## Predecir los N siguientes valores

#u_past_hist= conf["past_hist"]
#u_future_traget = conf["future_target"]
#
#train_split= ml_tools.data_split(data_u, 100)
#
#x_train, y_train = ml_tools.univariate_data(data_u, 0, train_split, u_past_hist, u_future_traget)
#
#model = Sequential()
#if conf["lstm2"] == 0:
#    model.add(LSTM(conf["lstm1"], activation =conf["act_func"], input_shape=x_train.shape[-2:]))
#    model.add(Dropout(conf["dropout"]))
#elif conf["lstm3"] == 0:
#    model.add(LSTM(conf["lstm1"], return_sequences=True, activation =conf["act_func"], input_shape=x_train.shape[-2:]))
#    #kernel_regularizer=tf.keras.regularizers.l2(conf["l2_reg"])
#    model.add(Dropout(conf["dropout"]))
#    model.add(LSTM(conf["lstm2"], activation =conf["act_func"]))
#    model.add(Dropout(conf["dropout"]))
#else:
#    model.add(LSTM(conf["lstm1"], return_sequences=True, activation =conf["act_func"], input_shape=x_train.shape[-2:]))
#    model.add(Dropout(conf["dropout"]))
#    model.add(LSTM(conf["lstm2"], return_sequences=True, activation =conf["act_func"]))
#    model.add(Dropout(conf["dropout"]))
#    model.add(LSTM(conf["lstm3"], activation =conf["act_func"]))
#    model.add(Dropout(conf["dropout"]))
#model.add(Dense(1))
#if conf["callbacks"] == 1:
#    print("using callbacks...") #tensorboard --logdir=logs/fit
#    log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
#    early_s = EarlyStopping('loss', patience = conf["early_s"], mode = 'min')
#    lr_red = ReduceLROnPlateau('loss', patince= conf["early_s"], mode = 'min', verbose= conf["early_s"])
#    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=conf["lr"]), loss=conf["loss"],metrics=[conf["metrics"]])
#    m_perf = model.fit(x_train, y_train, batch_size = conf["batch_size"], epochs= conf["epoch"], shuffle = False, validation_data = (x_val, y_val), callbacks=[early_s, lr_red, tensorboard])
#else:
#    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=conf["lr"]), loss=conf["loss"],metrics=[conf["metrics"]])
#    m_perf = model.fit(x_train, y_train, batch_size = conf["batch_size"], epochs= conf["epoch"], shuffle = False, validation_data = (x_val, y_val))
#
#graphs.plot_model_metric(m_perf, 'loss', save = True)
#
#n_ahead = 24
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
