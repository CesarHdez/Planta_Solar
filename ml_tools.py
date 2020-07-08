import numpy as np
import pandas as pd
import datetime
import os
import time
import shutil
import glob

import tools
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

def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step = False):
    data = []
    labels = []
    
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

def data_split(array, percent):
	limit = int(len(array) * percent / 100)
	return limit

def normaize(data_u):
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

def model_out_tunep(yhat):
	yhat[yhat < 0] = 0
	return yhat

def forecast_dataframe(data, yhat, n_ahead, hist_tail=300):
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

def save_experiment(multi_model=False):
    if not os.path.exists(settings.exp_path):
        os.mkdir(settings.exp_path)
    time_name = str(time.time())
    dir_name = settings.exp_path + time_name
    try:
        os.mkdir(dir_name)
    except:
        print("Directory Error")
    
    #graficos
    for pic in glob.glob(settings.g_path+"\\*.png"):
        shutil.copy2(pic, dir_name)
        
    #modelos
    for pic in glob.glob(settings.m_path+"\\*.h5"):
        shutil.copy2(pic, dir_name)
        
    #config
    if multi_model:
        shutil.copy2(settings.this_path+'lstm_config.json', dir_name)
    else:
        shutil.copy2(settings.this_path+'lstm_m_config.json', dir_name)
    
def clean_output_folders():
    if os.path.exists(settings.g_path):
        shutil.rmtree(settings.g_path)
    if os.path.exists(settings.m_path):
        shutil.rmtree(settings.m_path)
    try:
        os.mkdir(settings.g_path)
        os.mkdir(settings.m_path)
    except:
        print("Directory Error")