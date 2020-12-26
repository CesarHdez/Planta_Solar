import pandas as pd
import numpy as np
import json
import os

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
def get_lib_models():
    list_model = []
    list_config = []
    model_type = '_u'
    with os.scandir(settings.app_models_path) as folders:
        folders = [folders.name for folders in folders if folders.is_dir()]  
        
    for folder in folders:
        with os.scandir(settings.app_models_path +'/' +folder ) as files:
            files = [files.name for files in files if files.is_file() and files.name.endswith(model_type +'.h5')]
        model = load_model(settings.app_models_path +'/' + folder +'/' +files[0])
        list_model.append(model)
        with os.scandir(settings.app_models_path +'/' +folder ) as files:
            files = [files.name for files in files if files.is_file() and files.name.endswith('config.json')]
            with open(settings.app_models_path +'/' + folder +'/' +files[0]) as config_file:
                conf = json.load(config_file)
        list_config.append(conf)
    return list_config, list_model
#---------------------------------------------
def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    shared_keys = d1_keys.intersection(d2_keys)
    same = set(o for o in shared_keys if d1[o] == d2[o])
    return same
#---------------------------------------------
list_config, list_model = get_lib_models()
def find_model(list_config, list_model, query):
    foud_index=None
    for i in range(len(list_config)):
        if dict_compare(list_config[i], query) == set(query.keys()):
            foud_index=i
    if foud_index is None:
        return None
    else:
        return list_model[foud_index]
#---------------------------------------------
    
def get_data_2_predict(data, conf, date):
    #data= data[conf["y_var"]]
    date_time_str = date
    date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
    last_val = date_time_obj - datetime.timedelta(hours= 1)
    first_val =  last_val - datetime.timedelta(hours= conf["past_hist"])
    data = data.loc[first_val:last_val]
    data_r, data_mean, data_std = ml_tools.normalize(data[conf["y_var"]].values)
    data_r.reshape((len(data_r),1))
    return data, data_r, data_mean, data_std

#---------------------------------------------
def make_predicción(model, n_ahead, data, data_r, data_mean_2, data_std_2):
    data_r.reshape((len(data_r),1))
    yhat = ml_tools.predict_n_ahead(model, n_ahead, data_r)
    yhat = ml_tools.desnormalize(np.array(yhat), data_mean_2, data_std_2)
    yhat = ml_tools.model_out_tunep(yhat)
    #-----------------------------------------------
    graphs.plot_next_forecast(data, yhat, n_ahead, save=True)
    fc = ml_tools.forecast_dataframe(data, yhat, n_ahead)
    return fc
#----------------------------------------------------
def predicción_stats(data,fc):
    pass
#----------------------------------------------------
    

query = {
	"layer1" : 150,
	"act_func" : "LeakyReLU",
	"loss" : "logcosh",
	"optimizer": "RMSprop",
	"past_hist" : 48,
	"split_p" : 80
}

model = find_model(list_config, list_model, query)
if model is None:
    print("El modelo que requiere no se encuentra en la librería")
else:
    n_ahead = 24
    date = '2019-06-03 00:00:00'
    
    data, data_r, data_mean_2, data_std_2 = get_data_2_predict(data, conf, date)
    fc = make_predicción(model, n_ahead, data, data_r, data_mean_2, data_std_2)
    
    fc =fc.iloc[-49:]
    print(fc)
