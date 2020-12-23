import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import tools
import ml_tools
import settings

#--------------------------------
#LSTM
#--------------------------------

def create_time_steps(length):
  return list(range(-length, 0))

def show_plot(plot_data, delta, title, save=False):
	labels = ['History', 'True Future', 'Model Prediction']
	marker = ['.-', 'rx', 'go']
	time_steps = create_time_steps(plot_data[0].shape[0])
	if delta:
	  future = delta
	else:
	  future = 0

	plt.title(title)
	plt.figure(figsize=(12, 6))
	for i, x in enumerate(plot_data):
	  if i:
	    plt.plot(future, plot_data[i], marker[i], markersize=10,
	             label=labels[i])
	  else:
	    plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
	plt.legend()
	plt.xlim([time_steps[0], (future+5)*2])
	plt.xlabel('Time-Step')
	plt.grid()
	if save:
		plt.savefig(settings.g_path+title+'.png')
	plt.grid()
	plt.show()

def multi_step_plot(history, true_future, prediction, STEP, save=False):
	plt.figure(figsize=(12, 6))
	num_in = create_time_steps(len(history))
	num_out = len(true_future)
	title ='Multi Step Prediction'
	plt.plot(num_in, np.array(history), label='History')
	plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo', label='True Future')
	if prediction.any():
	  plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
	           label='Predicted Future')
	plt.legend(loc='upper left')
	plt.title(title)
	plt.grid()
	if save:
		plt.savefig(settings.g_path+title+'.png')
	plt.grid()
	plt.show() 	

def plot_model_learn_days(relat, save=False):
	fc = relat
	fc = fc.rename(columns={'ENERGY': 'Real','forecast':'Pronóstico'})
	plt.figure(figsize=(12, 6))
	title= 'Pronóstico diario'
	for dtype in ['Real', 'Pronóstico']:
	    plt.plot(
	        fc.index,
	        dtype,
	        '-',
	        data=fc,
	        label=dtype,
	        alpha=0.8
	    )
	plt.title(title)
	#plt.xlabel('Time')
	plt.xlabel('Tiempo')
	#plt.ylabel(y_var)
	plt.ylabel('Energía Diaria Exportada (KW)')
	plt.legend()
	plt.grid()
	if save:
		plt.savefig(settings.g_path+title+'.png')
	plt.grid()
	plt.show()



def plot_model_learn(data, yhat, y_var='ENERGY', save=False):
	fc = data.tail(len(yhat)).copy()
	#fc.reset_index(inplace=True)
	fc['forecast'] = yhat
	fc = fc[[y_var, 'forecast']]
	fc.columns=['actual','forecast']
	# Ploting the forecasts
	title= 'Model Rsults'
	plt.figure(figsize=(12, 6))
	for dtype in ['actual', 'forecast']:
	    plt.plot(
	        fc.index,
	        dtype,
	        '-',
	        data=fc,
	        label=dtype,
	        alpha=0.8
	    )
	plt.title(title)
	#plt.xlabel('Time')
	plt.xlabel('Tiempo')
	#plt.ylabel(y_var)
	plt.ylabel('Energía Exportada (KW)')
	plt.legend()
	plt.grid()
	if save:
		plt.savefig(settings.g_path+title+'.png')
	plt.grid()
	plt.show()

def plot_scatter_learn(data, yhat, save=False):
	fc = data.tail(len(yhat)).copy()
	real = fc['ENERGY'].values
	title= 'Scatter Model Results'
	plt.figure(figsize=(12, 6))
	plt.scatter(yhat, real)
	plt.plot(real, real, 'r--')
	plt.title(title)
	plt.xlabel('Forecast')
	plt.ylabel('Real')
	plt.legend()
	plt.grid()
	if save:
		plt.savefig(settings.g_path+title+'.png')
	plt.grid()
	plt.show()

def plot_scatter_learn_days(data, yhat, save=False):
	fc = data.tail(len(yhat)).copy()
	real = fc['ENERGY'].values
	title= 'Scatter Model Results_Daily'
	plt.figure(figsize=(12, 6))
	plt.scatter(yhat, real)
	plt.plot(real, real, 'r--')
	#plt.title(title)
	plt.xlabel('Pronóstico')
	plt.ylabel('Real')
	plt.legend()
	plt.grid()
	if save:
		plt.savefig(settings.g_path+title+'.png')
	plt.grid()
	plt.show()

def plot_scatter_learn_days_2(data, save=False):
	data = data.rename(columns={'ENERGY': 'real'})
	sns.set(rc={'figure.figsize':(12,6)})
	sns_plot = sns.jointplot("forecast", "real", data=data, kind='reg',)
	if save:
		sns_plot.savefig(settings.g_path+title+'.png')

def plot_next_forecast(data, yhat, n_ahead, hist_tail= 300, save=False):
	fc = ml_tools.forecast_dataframe(data, yhat, n_ahead, hist_tail)
	#fc.reset_index(inplace=True, drop=True)
	#
	# Ploting the forecasts
	title= 'Next ' + str(n_ahead) +' Forecast'
	plt.figure(figsize=(12, 6))
	for col_type in ['history', 'forecast']:
	    #print(col_type)
	    plt.plot(
	        'DateTime', 
	        'ENERGY', 
	        data=fc[fc['type']==col_type],
	        label=col_type
	        )
	plt.title(title)
	plt.xlabel('Time')
	plt.ylabel('ENERGY')
	plt.legend()
	plt.grid()
	if save:
		plt.savefig(settings.g_path+title+'.png')
	plt.grid()
	plt.show()


def plot_model_metric(m_perf, metric, save=False):
	if metric in m_perf.history:
		plt.figure(figsize=(12, 6))
		plt.plot(m_perf.history[metric])
		val_m = 'val_'+metric
		if val_m in m_perf.history:
			plt.plot(m_perf.history[val_m])
		title = 'Model '+ metric
		plt.title(title)
		plt.ylabel(metric)
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc='upper left')
		plt.grid()
		if save:
			plt.savefig(settings.g_path+title+'.png')
		plt.grid()
		plt.show()
	else:
		print('Metric no calculated')

