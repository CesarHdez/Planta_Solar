import numpy as np
import pandas as pd
import json

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.models import load_model
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import LSTM
#from tensorflow.keras.callbacks import TensorBoard

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM

import settings
import graphs
import ml_tools

ml_tools.clean_output_folders()

with open(settings.conf_path) as config_file:
    conf = json.load(config_file)

data = pd.read_excel(settings.ex_data, sheet_name='data')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)
data = data.astype(float)
#data = data['ENERGY']
data[conf["y_var"]].astype(float)
#data.plot(subplots = True)
#data_u = np.concatenate((data[out_var].values,np.flipud(data[out_var].values)))#duplicado y flipiado
data_u = data[conf["y_var"]].values


data_u, data_mean, data_std = ml_tools.normaize(data_u)

train_split= ml_tools.data_split(data_u, conf["split_p"])

u_past_hist= conf["past_hist"]
u_future_traget = conf["future_target"]

x_train, y_train = ml_tools.univariate_data(data_u, 0, train_split, u_past_hist, u_future_traget)
x_val, y_val = ml_tools.univariate_data(data_u, train_split, None, u_past_hist, u_future_traget)

model = Sequential()
model.add(LSTM(conf["lstm1"], input_shape=x_train.shape[-2:]))
model.add(Dense(1))
#tensorboard = TensorBoard(log_dir= ".logs/{}".format(time.time()))
model.compile(optimizer='adam', loss=conf["loss"], metrics=[conf["metrics"]])
#model.compile(optimizer='adam', loss=conf["loss"])
print(model.summary())
m_perf = model.fit(x_train, y_train, batch_size = conf["batch_size"], epochs= conf["epoch"], shuffle = False, validation_data = (x_val, y_val))
#m_perf = model.fit(x_train, y_train, batch_size = conf["batch_size"], epochs= conf["epoch"], shuffle = False, validation_data = (x_val, y_val), callbacks=[tensorboard])
model.save(settings.m_path+'lstm_u.h5')

graphs.plot_model_metric(m_perf, 'loss', save = True)

for i in conf["metrics"]:
    graphs.plot_model_metric(m_perf, i, save = True)

#model = load_model(settings.m_path+'lstm_u.h5')

yhat= model.predict(x_val)
#yhat = [y[0] for y in model.predict(x_val)]

it = 17
graphs.show_plot([x_val[it], y_val[it], yhat[it]], 0,'LSTM model', True)

yhat = ml_tools.desnormalize(yhat, data_mean, data_std)

graphs.plot_model_learn(data, yhat, True)

####################################
# Predecir los N siguientes valores


u_past_hist= conf["past_hist"]
u_future_traget = conf["future_target"]

train_split= ml_tools.data_split(data_u, 100)

x_train, y_train = ml_tools.univariate_data(data_u, 0, train_split, u_past_hist, u_future_traget)

model = Sequential()
model.add(LSTM(conf["lstm1"], input_shape=x_train.shape[-2:]))
model.add(Dense(1))
model.compile(optimizer='adam', loss=conf["loss"],  metrics=[conf["metrics"]])
m_perf = model.fit(x_train, y_train, batch_size = conf["batch_size"], epochs= conf["epoch"], shuffle = False)

graphs.plot_model_metric(m_perf, 'loss', save = True)

n_ahead = 24
n_ahead = conf["n_ahead"]
last_input= x_train[-1]

yhat = ml_tools.predict_n_ahead(model, n_ahead, last_input)

yhat = ml_tools.desnormalize(np.array(yhat), data_mean, data_std)

yhat = ml_tools.model_out_tunep(yhat)

graphs.plot_next_forecast(data, yhat, n_ahead, save=True)

fc = ml_tools.forecast_dataframe(data, yhat, n_ahead)
fc =fc.iloc[-49:]
print(fc)

ml_tools.save_experiment()
