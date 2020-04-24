#import tensorflow as tf
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sbn
#import sklearn
#import keras
import settings
import tools



#Obtener los nombres de los archivos para procesar
files_name = tools.ls2(settings.path)
#crear dataframe con los headers de acuerdo a la data
full_data = pd.DataFrame(columns= settings.headers_list)
#agregar al dataframe cada mes(archivo de excel)
for i in files_name:
    full_data =tools.add_data_month(full_data, settings.path +'/' + i)

#Dar formato al dataframe: setear fecha como indice y ordenar los datos cronológicamentes
full_data = tools.format_dataframe(full_data, settings.headers_list[0])
#cambiar valores de radiación solar negativos a zero
full_data = tools.neg_irrad_2_zero(full_data)
#cambiar los datos negativos de generación de energía a positivos
full_data = tools.negative_to_positive(full_data, 'ENERGY')

#obtener solo las horas de sol
full_data = tools.full_data_sun_hours(full_data, 'ENERGY')

#full_data contiene los datos con frecuencia de una hora.
#full_data en distintas frecuencias
diary = full_data.resample('D').mean()
weekly = full_data.resample('W').mean()
monthly = full_data.resample('M').mean()

#Gráficas
#---------
sbn.set(rc={'figure.figsize':(15, 5)})
full_data['ENERGY'].plot(linewidth=1)

#--------------------------------------------------------
#A partir de  aqui minería de datos
#--------------------------------------------------------



