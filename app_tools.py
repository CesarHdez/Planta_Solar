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
date = '2019-06-03 00:00:00'
def get_data_2_predict(data, conf, date):
    data= data[conf["y_var"]]
    date_time_str = date
    date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
    last_val = date_time_obj - datetime.timedelta(hours= 1)
    first_val =  last_val - datetime.timedelta(hours= conf["past_hist"])
    data_r = data.loc[first_val:last_val].values
    data_r, data_mean, data_std = ml_tools.normalize(data_r)
    data_r.reshape((len(data_r),1))
    return data_r, data_mean, data_std

data_r, data_mean_2, data_std_2 = get_data_2_predict(data, conf, date)
#---------------------------------------------


model = load_model(settings.m_path+conf['type']+'_u'+'.h5')
#model = load_model('Custom1_u.h5')
#model = load_model('Custom1_u.h5')
n_ahead = conf["n_ahead"]
data_r.reshape((len(data_r),1))
yhat = ml_tools.predict_n_ahead(model, n_ahead, data_r)

yhat = ml_tools.desnormalize(np.array(yhat), data_mean_2, data_std_2)

yhat = ml_tools.model_out_tunep(yhat)

#-----------------------------------------------

graphs.plot_next_forecast(data, yhat, n_ahead, save=True)

fc = ml_tools.forecast_dataframe(data, yhat, n_ahead)
fc =fc.iloc[-49:]
print(fc)
