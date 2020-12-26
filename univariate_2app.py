
"""
Univariate 2 app
"""
import pandas as pd
import numpy as np
import json

from keras.models import load_model

import model_mk
import settings
import graphs
import ml_tools
import tools
import datetime
from tensorflow.keras.models import load_model

with open(settings.conf_u_path) as config_file:
    conf = json.load(config_file)

data = pd.read_excel(settings.ex_data, sheet_name='data')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)
#data = data[:-120]
data = data.astype(float)
#data = data['ENERGY']
data[conf["y_var"]].astype(float)

#---------------------------------------------
data_u = data[conf["y_var"]].values

data_u, data_mean, data_std = ml_tools.normalize(data_u)

#Predecir los N siguientes valores

#Aqui comenzaria el for para generar modelos
#-----------------------------------------------
u_past_hist= conf["past_hist"]
u_future_traget = conf["future_target"]

train_split= ml_tools.data_split(data_u, 100)

x_train, y_train = ml_tools.univariate_data(data_u, 0, train_split, u_past_hist, u_future_traget)

#-----------------------------------------------


model, m_perf = model_mk.model_maker(conf, x_train, y_train)
model.save(settings.m_path+conf['type']+'_u'+'.h5')
graphs.plot_model_metric(m_perf, 'loss', save = False)
#
#
ml_tools.save_model_2app(conf)

#Aqui termina el for para generar modelos


#-----------------------------------------------
n_ahead = conf["n_ahead"]
last_input= x_train[-1]
#-----------------------------------------------


yhat = ml_tools.predict_n_ahead(model, n_ahead, last_input)

yhat = ml_tools.desnormalize(np.array(yhat), data_mean, data_std)

yhat = ml_tools.model_out_tunep(yhat)

#-----------------------------------------------

graphs.plot_next_forecast(data, yhat, n_ahead, save=True)

fc = ml_tools.forecast_dataframe(data, yhat, n_ahead)
fc =fc.iloc[-49:]
print(fc)

