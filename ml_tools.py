import numpy as np
import pandas as pd
import datetime
import os
import time
import shutil
import glob
import json
import pickle

import settings
#-----------------
#With data
#-----------------

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None or end_index > (len(dataset) - target_size):
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        #Reshape data to (history, 1)
        data. append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)

def univariate_data_2(dataset_x, dataset_y, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None or end_index > (len(dataset_x) - target_size):
        end_index = len(dataset_x) - target_size
    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        #Reshape data to (history, 1)
        data. append(np.reshape(dataset_x[indices], (history_size, 1)))
        labels.append(dataset_y[i+target_size])
    return np.array(data), np.array(labels)


def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step = False, future= False):
    data = []
    labels = []
    
    if future:
        start_index = start_index
        if end_index is None or end_index > (len(dataset) - target_size - history_size):
            end_index = len(dataset) - target_size - history_size
            
        for i in range(start_index, end_index):
            indices = range(i, i+history_size, step)
            data.append(dataset[indices])
            
            if single_step:
                labels.append(target[i+target_size])
            else:
                labels.append(target[i:i+target_size])
    else:
        start_index = start_index + history_size
        if end_index is None or end_index > (len(dataset) - target_size):
            end_index = len(dataset) - target_size
            
        for i in range(start_index, end_index):
            indices = range(i-history_size, i, step)
            data.append(dataset[indices])
            
            if single_step:
                labels.append(target[i+target_size])
            else:
                labels.append(target[i:i+target_size])
    return np.array(data), np.array(labels)

def multivariate_data_2(dataset, start_index, end_index, history_size, target_size, step, single_step = False):
    data = []
    
    start_index = start_index + history_size
    if end_index is None or end_index > (len(dataset) - target_size):
        end_index = len(dataset) - target_size
        
    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])
    return np.array(data)

def data_split(array, percent):
	limit = int(len(array) * percent / 100)
	return limit

def normalize_2(data_u):
    data_max = data_u.max(axis=0)
    data_min = data_u.min(axis=0)
    data_dif = data_max - data_min
    data_u = (data_u - data_min)/data_dif
    return data_u, data_min, data_dif

def normalize(data_u):
	data_mean = data_u.mean(axis=0)
	data_std = data_u.std(axis=0)
	data_u = (data_u - data_mean)/data_std
	return data_u, data_mean, data_std

#-----------------
#With the results
#-----------------

def desnormalize(data, mean, std):
    des_norm = (data*std)+mean
    return des_norm

def desnormalize_2(data, data_min, data_dif):
    des_norm = (data*data_dif) + data_min
    return des_norm

def model_out_tunep(yhat):
    yhat[yhat < 0] = 0
    ref = float(yhat[yhat < 9500].max())
    yhat[yhat > 9500] = ref
    return yhat

def forecast_relation(data, yhat):
	fc = data.tail(len(yhat)).copy()
	fc['forecast'] = yhat
	fc = fc[['ENERGY', 'forecast']]
	return fc

def forecast_dataframe(data, yhat, n_ahead, y_var='ENERGY', hist_tail=300):
	fc = data.tail(hist_tail).copy() 
	fc['type'] = 'history'
	fc = fc[['ENERGY', 'type']]

	#
	last_date = fc.index[-1]
	hat_frame = pd.DataFrame({
	    'DateTime': [last_date + datetime.timedelta(hours=x + 1) for x in range(n_ahead)], 
	    'ENERGY': yhat,
	    'type': 'forecast'
	})
	hat_frame['DateTime']=pd.to_datetime(hat_frame['DateTime'])
	hat_frame.set_index('DateTime', inplace=True)
	fc = fc.append(hat_frame)
	#fc = tools.negative_to_zero(fc, 'ENERGY')
	fc ['DateTime'] = fc.index
	fc.reset_index(inplace=True, drop=True)
	return fc

def forecast_dataframe_2(data, yhat, n_ahead, y_var='ENERGY', hist_tail=300):
    fc = data.tail(hist_tail).copy() 
    fc['type'] = 'history'
    fc = fc[[y_var, 'type']]

    #
    last_date = fc.index[-1]
    hat_frame = pd.DataFrame({
        'DateTime': [last_date + datetime.timedelta(hours=x + 1) for x in range(n_ahead)], 
        y_var: yhat,
        'type': 'forecast'
    })
    hat_frame['DateTime']=pd.to_datetime(hat_frame['DateTime'])
    hat_frame.set_index('DateTime', inplace=True)
    fc = fc.append(hat_frame)
    #fc = tools.negative_to_zero(fc, 'ENERGY')
    fc ['DateTime'] = fc.index
    fc.reset_index(inplace=True, drop=True)
    return fc

def predict_n_ahead(model, n_ahead, last_input):
    X = last_input
    X = np.reshape(X, (1, len(X), 1))
    #yhat = []
    
    yhat = model.predict(X)
    
    yhat = []
    
    for _ in range(n_ahead):
        # Making the prediction
        fc = model.predict(X)
        yhat.append(fc)
    
        # Creating a new input matrix for forecasting
        X = np.append(X, fc)
    
        # Ommiting the first variable
        X = np.delete(X, 0)
    
        # Reshaping for the next iteration
        X = np.reshape(X, (1, len(X), 1))
    
    yhat = [y[0][0] for y in yhat]
    return yhat


#-----------------
#With experiments
#-----------------

def load_perf(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def save_perf(name, perf):
    f = open(name,"wb")
    pickle.dump(perf,f)
    f.close()

def save_experiment(conf, multi_model=False):
    if not os.path.exists(settings.exp_path):
        os.mkdir(settings.exp_path)
    time_name = str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
    dir_name = settings.exp_path + time_name
    try:
        os.mkdir(dir_name)
    except:
        print("Directory Experiment Error")
    
    #graficos
    for pic in glob.glob(settings.g_path+"*.png"):
        shutil.copy2(pic, dir_name)
        
    #modelos
    for pic in glob.glob(settings.m_path+"*.h5"):
        shutil.copy2(pic, dir_name)
        
    for pic in glob.glob(settings.m_path+"*.pkl"):
        shutil.copy2(pic, dir_name)
        
    for pic in glob.glob(settings.m_path+"*.xlsx"):
        shutil.copy2(pic, dir_name)
    #config
    if multi_model:
        shutil.copy2(settings.this_path+'m_config.json', dir_name)
    else:
        shutil.copy2(settings.this_path+'u_config.json', dir_name)

    #model code
    shutil.copy2(settings.mk_path+conf["type"]+'.py', dir_name)

def save_model_2app(conf, multi_model=False):
    if not os.path.exists(settings.exp_path):
        os.mkdir(settings.exp_path)
    time_name = str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
    dir_name = settings.app_models_path + time_name
    try:
        os.mkdir(dir_name)
    except:
        print("Directory Experiment Error")
        
    #modelos
    for pic in glob.glob(settings.m_path+"*.h5"):
        shutil.copy2(pic, dir_name)
        
    #config
    if multi_model:
        shutil.copy2(settings.this_path+'m_config.json', dir_name)
    else:
        shutil.copy2(settings.this_path+'u_config.json', dir_name)

    #model code
    shutil.copy2(settings.mk_path+conf["type"]+'.py', dir_name)
    
def clean_output_folders():
    if os.path.exists(settings.g_path):
        shutil.rmtree(settings.g_path)
    if os.path.exists(settings.m_path):
        shutil.rmtree(settings.m_path)
    try:
        os.mkdir(settings.g_path)
        os.mkdir(settings.m_path)
    except:
        print("Directory Clean Error")


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))


def get_model_stats(m_perf, it = -1, val = False):
    stats_d = dict()
    metrics = ['mse', 'mae', 'mape', 'root_mean_squared_error']
    if val :
        for m in metrics:
            stats_d['val_'+m] = m_perf['val_'+m][it]
    else:
        for m in metrics:
            stats_d[m] = m_perf[m][it]
    return stats_d
