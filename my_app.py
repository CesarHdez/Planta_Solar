import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt
import os
import streamlit_theme as stt
import app_tools
import settings
from keras.models import load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential



os.environ['TZ'] = 'UTC'
@st.cache
def load_data(data=None):
	if data is None:
		data = pd.read_excel('app_data_2.xlsx', sheet_name='data')
	else:
		data = pd.read_excel(data, sheet_name='data')
	data['DateTime'] = pd.to_datetime(data['DateTime'])
	data.set_index('DateTime', inplace=True)
	data = data.astype(float)
	#data = data.loc['01-01-2020':'02-28-2020']
	# data = data[-2160:]
	return data
	
def load_models():
	list_config, list_model = app_tools.get_lib_models()
	return list_config, list_model

@st.cache
def load_models_config():
	list_config, folders = app_tools.get_lib_models_config()
	return list_config, folders

# def load_models_pkl():
# 	list_config, list_model = app_tools.get_lib_models()
# 	return list_config, list_model

#stt.set_theme({'primary': '#ffffff'})
st.title('Predictor de Producción Fotovoltaica')
st.image(settings.app_src_path+"solar_plant_logo.png", use_column_width=True)

load_state = st.empty()
date_state = st.empty()
load_state.text("")


st.sidebar.header('Datos')
upload_file = st.sidebar.file_uploader("Cargue el archivo de datos (.csv o .xlsx)", type=["csv","xlsx"])
#--------------------------------------------------------------------------

st.sidebar.header('Parámetros del Modelo')
def model_parameters():
    neurons = st.sidebar.selectbox('Número de Neuronas',(50,100,150))
    act_func = st.sidebar.selectbox('Función de Activación',('relu','elu','LeakyReLU'))
    loss = st.sidebar.selectbox('Función de Pérdida',('MSE','huber_loss','logcosh'))
    optmz = st.sidebar.selectbox('Optimizador',('Adam','RMSprop'))
    input_size = st.sidebar.selectbox('Tamaño de la entrada',(24, 48, 72))
    query = {'layer1': neurons,
            'act_func': act_func,
            'loss': loss,
            'optimizer': optmz,
            'past_hist':input_size}
    features = pd.DataFrame(query, index=[0])
    features = features.rename(columns = {'layer1':'Número de Neuronas',
    	'act_func':'Función de Activación','loss':'Función de Pérdida',
    	'optimizer':'Optimizador', 'past_hist':'Tamaño de la entrada'})
    return query, features
#----------------------------------------------------------------------------

with st.beta_expander("Sobre los modelos"):
    st.write("""Los modelos que utiliza esta aplicación comparten la misma arquitectura
    que consiste en una Red Neuronal Artificial con dos capas ocultas: la primera conformada 
    por celdas recurrentes y la segunda por neuronas convencionales. El número de neuronas 
    recurrentes (paámetro a configurar) es el doble del número de neuronas de la segunda capa.""")
    st.image(settings.app_src_path+"model_structure.png")


query, features= model_parameters()
st.subheader('Parámetros del Modelo')
st.table(features.assign(hack='').set_index('hack'))
#----------------------------------------------------------------------------

st.sidebar.header('Sobre la predicción')
min_date = dt.datetime.strptime('2020-03-04', '%Y-%m-%d')
print(type(min_date))
# date = st.sidebar.date_input('Día a predecir', data.index.get_level_values(0)[0])
date_comp = st.sidebar.empty()
date = date_comp.date_input('Día a predecir', min_date)
time = st.sidebar.time_input('A partir de las', dt.time(00, 00))
if time.minute !=0:
	time= dt.time(time.hour, 00)
n_ahead = st.sidebar.slider('Horas a predecir', 1, 24, 1)
datetime_str = str(date)+' '+str(time)
date_state = st.empty()
date_state.info("Se realizará una predicción de {} hora(s) del día {} a partir de las {}".format(n_ahead, date, time))

#----------------------------------------------------------------------------

pressed = st.button('Buscar Modelo & Predecir')
if pressed:
    load_state.warning('Cargando datos...')
    if upload_file is not None:
    	data = load_data(upload_file)
    else:
    	data = load_data()
    # print(data)
    # st.header("Datos")
    # st.dataframe(data)
    # st.write(data.first_valid_index())
    # st.write(data.last_valid_index())
    valid_date = data.first_valid_index() + dt.timedelta(hours= query['past_hist'])
    # date_comp.date_input('Día a predecir', valid_date)
    datetime_w = dt.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    if datetime_w < valid_date or datetime_w > data.last_valid_index():
    	date_state.warning("La fecha a predecir es incorrecta para los datos cargados. Se tomará por defecto {}".format(valid_date))
    	datetime_str = valid_date.strftime("%Y-%m-%d %H:%M:%S")
    else:
        date_state.info("Se realizó una predicción de {} hora(s) del día {} a partir de las {}".format(n_ahead, date, time))
    data_master = data
    load_state.warning('Cargando modelos...')
    # list_config, list_model = load_models()
    list_config, folders = load_models_config()
    load_state.success('Carga completada!')
    #----------------------------------------------------------------------------
    # model, conf = app_tools.find_model(list_config, list_model, query)
    conf, model_index = app_tools.find_model_by_config(list_config, query)
    if type(conf)==list:
        st.error("No se encontró un modelo con estas características en el banco de modelos")
        flag_input_show = False
    else:
        model = app_tools.get_model_by_folderindex(folders, model_index)
        st.success("Modelo cargado correctamente")
        data, data_r, data_mean_2, data_std_2= app_tools.get_data_2_predict(data,conf, datetime_str)
        st.subheader('Datos seleccionados para la entrada del modelo')
        left_column, right_column = st.beta_columns([1.5,1])
        left_column.text("KWh [Energía Activa Exportada,(EAE)]")
        left_column.line_chart(data.rename(columns = {'ENERGY':'EAE'})['EAE'])
        right_column.dataframe(data.rename(columns = {'ENERGY':'EAE'})['EAE'])

        predict_header = st.empty()

        predict_header.info('Realizando predicción...')
        fc = app_tools.make_predicción(model, n_ahead, data, data_r, data_mean_2, data_std_2)

        comp_df = app_tools.compare_df(data_master,fc)
        comp = comp_df.drop(columns=['type'])

        left_column_2, right_column_2 = st.beta_columns(2)
        predict_header.subheader('Resultados de la Predicción:')
        left_column_2.dataframe(comp[-n_ahead:].apply(pd.Series.round))
        stats_df = app_tools.comp_stats(comp_df)
        right_column_2.dataframe(stats_df.apply(pd.Series.round).T)
        st.markdown('***Gráfico de Predicción***')
        st.text("KWh")
        st.line_chart(comp)


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer:after {
            content:"Esta aplicación fue desarrollada como parte del trabajo de investigación de César Hernández para optar por el grado de Magister en Informátiga y Ciencias de la Computación de la Universidad de Atacama."; 
            visibility: visible;
            display: block;
            position: center;
            top: 2px;
            }
            </style>
            """
#"This app was developed as part of the César Hernández's research work to get the Master's degree in Informatics and Computer Science from Atacama University, Chile."
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
		
	#     st.vega_lite_chart(comp[-n_ahead:], {
	#     'mark': {'type': 'circle', 'tooltip': True},
	#     'encoding': {
	#         'x': {'field': 'Predicción', 'type': 'quantitative'},
	#         'y': {'field': 'Real', 'type': 'quantitative'},
	#     },
	# })

# st.line_chart(data[start_date:end_date]['WS1'])
# st.line_chart(data[start_date:end_date]['IRRAD1'])
# st.line_chart(data[start_date:end_date]['TEMP1'])
# st.line_chart(data[start_date:end_date]['WANG'])

