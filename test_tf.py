#import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import LabelEncoder
#from sklearn.neural_network import MLPRegressor

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Softmax
#from keras.layers import LSTM

#from sklearn import metrics
#import math
#import keras

import settings
import tools
#import ml_tools


def data_prep():	
    files_name = tools.ls2(settings.path)
    full_data= pd.DataFrame()
    tools.printProgressBar(0,len(files_name),prefix = 'Progress:', suffix = 'Complete', length = 30)
    count=0
    for i in files_name:
        count= count+1
        tools.printProgressBar(count, len(files_name),prefix = 'Progress:', suffix = 'Complete', length = 30)
        #print(count,"/",len(files_name))
        full_data =tools.add_data_month(full_data, settings.path +'/' + i)
    full_data = full_data.sort_index(axis=0)
    full_data = tools.neg_irrad_2_zero(full_data)
    full_data = tools.negative_to_positive(full_data, 'ENERGY')
    full_data = tools.change_outliers_values(full_data, 'ENERGY')
    #full_data = tools.full_data_sun_hours(full_data, 'ENERGY')
    #full_data = tools.delete_cols(full_data, settings.cols2delete)
    full_data = full_data.astype(float)
    return full_data

full_data = data_prep()

#--------------------------------
#adding jan
#to_add = pd.read_excel('enero.xlsx', sheet_name='data')
#to_add['DateTime'] = pd.to_datetime(to_add['DateTime'])
#to_add.set_index('DateTime', inplace=True)
#
#full_data = pd.concat([full_data, to_add])
#full_data = full_data.sort_index(axis=0)
#-------------------------------

full_data.to_excel('full_data.xlsx', sheet_name='data')
#
#full_data contiene los datos con frecuencia de una hora.
#full_data en distintas frecuencias
daily = full_data.resample('D').mean()
weekly = full_data.resample('W').mean()
monthly = full_data.resample('M').mean()

#Análisis de Correlación


#dataset con el que se va a trabaja
#full_data = full_data.rename(columns={'ENERGY': 'EAE'})
dataset = full_data
#dataset = dataset.loc['08-29-2020':'09-5-2020']

f, ax = plt.subplots(figsize=(10, 8))
cor = dataset.astype(float).corr(method = 'pearson')
print(cor)

##------------------------------------------------
##Gráficos
##------------------------------------------------
#
#
#grafico para series temporales
sbn.set(rc={'figure.figsize':(10, 5)})
dataset['ENERGY'].plot(linewidth=1)
#
##graficar todas por separado
values = dataset.values
# specify columns to plot
#groups = [1, 3, 8, 10]
groups = [1, 2, 3, 4]
#fields = ['EAE','IRRAD1','IRRAD2','IRRAD3','IRRAD4','IRRAD5','TEMP1','TEMP2','WS1','WS2', 'WANG']
fields = ['ENERGY','IRRAD1','IRRAD2','IRRAD3','IRRAD4','IRRAD5','TEMP1','TEMP2','WS1','WS2', 'WANG']
i = 1
# plot each column
plt.figure(figsize=(25, 30))
for j in fields:
	plt.subplot(len(fields), 1, i)
	plt.plot(dataset[j],label=j)
	plt.title(j, y=0.5, loc='left')
	i += 1
plt.show()
