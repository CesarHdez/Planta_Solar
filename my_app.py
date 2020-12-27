import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt
import os
import streamlit_theme as stt
import app_tools


os.environ['TZ'] = 'UTC'

@st.cache
def load_data():
	data = pd.read_excel('full_data.xlsx', sheet_name='data')
	data['DateTime'] = pd.to_datetime(data['DateTime'])
	data.set_index('DateTime', inplace=True)
	data = data.astype(float)
	data = data[:200]
	return data
	
# @st.cache
def load_models():
	list_config, list_model = app_tools.get_lib_models()
	return list_config, list_model

#stt.set_theme({'primary': '#ffffff'})
st.title('Predictor de Producción Fotovoltaica')

load_state = st.text('Cargando datos y modelos...')

data = load_data()
list_config, list_model = load_models()

load_state = st.text('Carga completada!')



st.sidebar.header('Parámetros del Modelo')
def model_parameters():
    neurons = st.sidebar.selectbox('Número de Neuronas',(50,100,150,200))
    act_func = st.sidebar.selectbox('Función de Activación',('relu','elu','selu','LeakyReLU'))
    loss = st.sidebar.selectbox('Función de Pérdida',('mse','mae','huber_loss','logcosh'))
    optmz = st.sidebar.selectbox('Optimizador',('Adam','RMSprop','Adagrad','SGD'))
    input_size = st.sidebar.selectbox('Tamaño de la entrada',(24, 48, 72, 96))
    query = {'layer1': neurons,
            'act_func': act_func,
            'loss': loss,
            'optimizer': optmz,
            'past_hist':input_size}
    features = pd.DataFrame(query, index=[0])
    return query, features


query, features= model_parameters()
st.subheader('Parámetros del Modelo')
st.write(features)


# print(data)
# st.header('Data Fixed')
# st.dataframe(data)



st.sidebar.header('Sobre la predicción')
min_date = dt.datetime.strptime('2019-06-03', '%Y-%m-%d')
# date = st.sidebar.date_input('Día a predecir', data.index.get_level_values(0)[0])
date = st.sidebar.date_input('Día a predecir', min_date,min_value=min_date)
time = st.sidebar.time_input('A partir de las', dt.time(00, 00))
n_ahead = st.sidebar.slider('Horas a predecir', 1, 24, 1)
datetime_str = str(date)+' '+str(time)

st.subheader('Datos de Entrada')
st.dataframe(data['ENERGY'])
st.line_chart(data['ENERGY'])


# model, conf = app_tools.find_model(list_config, list_model, query)
# if model is None:
#     st.text(datetime_str)
# else:
#     # n_ahead = 24
#     # date = '2019-06-03 00:00:00'
    
# 	data, data_r, data_mean_2, data_std_2= app_tools.get_data_2_predict(data,conf, datetime_str)

# 	fc = app_tools.make_predicción(model, n_ahead, data, data_r, data_mean_2, data_std_2)
# 	fc =fc.iloc[-49:]
# 	print(fc)
# 	st.dataframe(fc)



# st.line_chart(data[start_date:end_date]['WS1'])
# st.line_chart(data[start_date:end_date]['IRRAD1'])
# st.line_chart(data[start_date:end_date]['TEMP1'])
# st.line_chart(data[start_date:end_date]['WANG'])

