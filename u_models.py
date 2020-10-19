import numpy as np
import pandas as pd
import json
import datetime

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

ml_tools.clean_output_folders()

with open(settings.conf_u_path) as config_file:
    conf = json.load(config_file)

data = pd.read_excel(settings.ex_data, sheet_name='data')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)
data = data[:-140]
data = data.astype(float)
#data = data['ENERGY']
#data[conf["y_var"]].astype(float)
#data.plot(subplots = True)
#data_u = np.concatenate((data[out_var].values,np.flipud(data[out_var].values)))#duplicado y flipiado
data_u = data[conf["y_var"]].values


data_u, data_mean, data_std = ml_tools.normalize(data_u)

####################################
# Predecir los N siguientes valores

def get_vars_u_forecast(data ,conf):
    data_u = data[conf["y_var"]].values
    data_u, data_mean, data_std = ml_tools.normalize(data_u)
    u_past_hist= conf["past_hist"]
    u_future_traget = conf["future_target"]
    
    train_split= ml_tools.data_split(data_u, 100)
    
    x_train, y_train = ml_tools.univariate_data(data_u, 0, train_split, u_past_hist, u_future_traget)
    
    model, m_perf = model_mk.model_maker(conf, x_train, y_train)
    
    graphs.plot_model_metric(m_perf, 'loss', save = True)
    
    n_ahead = conf["n_ahead"]
    last_input= x_train[-1]
    
    yhat = ml_tools.predict_n_ahead(model, n_ahead, last_input)
    
    yhat = ml_tools.desnormalize(np.array(yhat), data_mean, data_std)
    
    if conf["y_var"] == 'ENERGY' or conf["y_var"] == 'IRRAD1':
        yhat = ml_tools.model_out_tunep(yhat)
    #yhat = ml_tools.model_out_tunep(yhat)
    
    #graphs.plot_next_forecast(data, yhat, n_ahead, save=True)
    
    fc = ml_tools.forecast_dataframe_2(data, yhat, n_ahead, conf["y_var"])
    return fc[-25:]

#(conf["n_ahead"]+1)
    
par_list = ["ENERGY","WS1", "TEMP1", "IRRAD1"]
first_time_flag = True


for i in par_list:
    conf["y_var"]=i   
    fc = get_vars_u_forecast(data, conf).iloc[:,[2,0]]
    fc.set_index("DateTime", inplace=True)
    if first_time_flag:
        data_2_append = fc
        first_time_flag = False
    else:
        data_2_append = pd.merge(data_2_append, fc, left_index=True, right_index=True)


#conf["y_var"]="





