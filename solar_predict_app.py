import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt
import time
import os
import app_tools
import settings
from keras.models import load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential


from streamlit.hashing import _CodeHasher

try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server
#-------------------------------------------------------
class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

@st.cache
def load_data(data=None):
	if data is None:
		data = pd.read_excel(settings.app_src_path +'app_data_2.xlsx', sheet_name='data')
	else:
		data = pd.read_excel(data, sheet_name='data')
	data['DateTime'] = pd.to_datetime(data['DateTime'])
	data.set_index('DateTime', inplace=True)
	data = data.astype(float)
	#data = data.loc['01-01-2020':'02-28-2020']
	# data = data[-2160:]
	return data

@st.cache
def load_models_config():
	list_config, folders = app_tools.get_lib_models_config()
	return list_config, folders
#--------------------------------------------------------------------------
def main():
    state = _get_state()
    pages = {"Español": page_spanish, "English": page_english,}

    page = st.sidebar.radio("Idioma/Language", tuple(pages.keys()))
    # leng_comp = st.sidebar.empty()
    # leng = leng_comp.radio("Idioma",('Español', 'English'))
    pages[page](state)
    state.sync()

def page_spanish(state):
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

    with st.beta_expander("Acerca de los modelos"):
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
        list_config, folders = load_models_config()
        load_state.success('Carga completada!')
        #----------------------------------------------------------------------------
        conf, model_index = app_tools.find_model_by_config(list_config, query)
        model_foud_state = st.empty()
        if type(conf)==list:
            model_foud_state.error("No se encontró un modelo con estas características en el banco de modelos")
            flag_input_show = False
        else:
            model = app_tools.get_model_by_folderindex(folders, model_index)
            model_foud_state.success("Modelo cargado correctamente")
            data, data_r, data_mean_2, data_std_2= app_tools.get_data_2_predict(data,conf, datetime_str)
            st.subheader('Datos seleccionados para la entrada del modelo')
            left_column, right_column = st.beta_columns([1.5,1])
            left_column.text("KWh [Energía Activa Exportada,(EAE)]")
            left_column.line_chart(data.rename(columns = {'ENERGY':'EAE'})['EAE'])
            right_column.dataframe(data.rename(columns = {'ENERGY':'EAE'})['EAE'])

            predict_header = st.empty()

            predict_header.info('Realizando predicción...')
            fc = app_tools.make_prediccion(model, n_ahead, data, data_r, data_mean_2, data_std_2)

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
            model_foud_state.empty()
            load_state.empty()


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


#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

def page_english(state):
    st.title('Photovoltaic Production Predictor')
    st.image(settings.app_src_path+"solar_plant_logo.png", use_column_width=True)

    load_state = st.empty()
    date_state = st.empty()
    load_state.text("")

    st.sidebar.header('Data')
    upload_file = st.sidebar.file_uploader("", type=["csv","xlsx"])
    #--------------------------------------------------------------------------

    st.sidebar.header('Model Parameters')
    def model_parameters():
        neurons = st.sidebar.selectbox('Number of Neurons',(50,100,150))
        act_func = st.sidebar.selectbox('Activation Function',('relu','elu','LeakyReLU'))
        loss = st.sidebar.selectbox('Loss Function',('MSE','huber_loss','logcosh'))
        optmz = st.sidebar.selectbox('Optimizer',('Adam','RMSprop'))
        input_size = st.sidebar.selectbox('Input Size',(24, 48, 72))
        query = {'layer1': neurons,
                'act_func': act_func,
                'loss': loss,
                'optimizer': optmz,
                'past_hist':input_size}
        features = pd.DataFrame(query, index=[0])
        features = features.rename(columns = {'layer1':'Number of Neurons',
            'act_func':'Activation Function','loss':'Loss Function',
            'optimizer':'Optimizer', 'past_hist':'Input Size'})
        return query, features
    #----------------------------------------------------------------------------

    with st.beta_expander("About models"):
        st.write("""The models used by this application share the same architecture 
            consisting of an Artificial Neural Network with two hidden layers: the 
            first made up of recurrent cells and the second made up of conventional 
            neurons. The number of recurrent neurons (parameter to configure) is 
            double number of neurons in the second layer.""")
        st.image(settings.app_src_path+"model_structure_en.png")


    query, features= model_parameters()
    st.subheader('Model Parameters')
    st.table(features.assign(hack='').set_index('hack'))
    #----------------------------------------------------------------------------

    st.sidebar.header('About prediction')
    min_date = dt.datetime.strptime('2020-03-04', '%Y-%m-%d')
    date_comp = st.sidebar.empty()
    date = date_comp.date_input('Day to predict', min_date)
    time = st.sidebar.time_input('Starting at', dt.time(00, 00))
    if time.minute !=0:
        time= dt.time(time.hour, 00)
    n_ahead = st.sidebar.slider('Hours to predict', 1, 24, 1)
    datetime_str = str(date)+' '+str(time)
    date_state = st.empty()
    date_state.info("A {} hour(s) prediction will be made on {} starting at {}".format(n_ahead, date, time))

    #----------------------------------------------------------------------------

    pressed = st.button('Find Model & Predict')
    if pressed:
        load_state.warning('Loading Data...')
        if upload_file is not None:
            data = load_data(upload_file)
        else:
            data = load_data()

        valid_date = data.first_valid_index() + dt.timedelta(hours= query['past_hist'])
        # date_comp.date_input('Día a predecir', valid_date)
        datetime_w = dt.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        if datetime_w < valid_date or datetime_w > data.last_valid_index():
            date_state.warning("The date to predict is incorrect for the uploaded data. It will be taken by default at {}".format(valid_date))
            datetime_str = valid_date.strftime("%Y-%m-%d %H:%M:%S")
        else:
            date_state.info("A {} hour(s) prediction was made on {} starting at {}".format(n_ahead, date, time))
        data_master = data
        load_state.warning('Loading Models...')
        list_config, folders = load_models_config()
        load_state.success('Loading Completed!')
        #----------------------------------------------------------------------------
        conf, model_index = app_tools.find_model_by_config(list_config, query)
        model_foud_state = st.empty()
        if type(conf)==list:
            model_foud_state.error("A model with these characteristics was not found in the model bank")
            flag_input_show = False
        else:
            model = app_tools.get_model_by_folderindex(folders, model_index)
            model_foud_state.success("Model loaded correctly")
            data, data_r, data_mean_2, data_std_2= app_tools.get_data_2_predict(data,conf, datetime_str)
            st.subheader('Selected data for model input')
            left_column, right_column = st.beta_columns([1.5,1])
            left_column.text("KWh [Exported Active Energy,(EAE)]")
            left_column.line_chart(data.rename(columns = {'ENERGY':'EAE'})['EAE'])
            right_column.dataframe(data.rename(columns = {'ENERGY':'EAE'})['EAE'])

            predict_header = st.empty()
            predict_header.info('Making prediction...')
            fc = app_tools.make_prediccion(model, n_ahead, data, data_r, data_mean_2, data_std_2)
            comp_df = app_tools.compare_df(data_master,fc)
            comp_df2 = comp_df.rename(columns = {'Predicción':'Prediction'})
            comp = comp_df2.drop(columns=['type'])

            left_column_2, right_column_2 = st.beta_columns(2)
            predict_header.subheader('Prediction Results:')
            left_column_2.dataframe(comp[-n_ahead:].apply(pd.Series.round))
            stats_df = app_tools.comp_stats(comp_df)
            stats_df = stats_df.apply(pd.Series.round).T
            right_column_2.dataframe(stats_df.rename(columns = {'Predicción':'Prediction'}))
            st.markdown('***Prediction Chart***')
            st.text("KWh")
            st.line_chart(comp)
            model_foud_state.empty()
            load_state.empty()

    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer:after {
                content:"This app was developed as part of the César Hernández's research work to get the Master's degree in Informatics and Computer Science from Atacama University, Chile."; 
                visibility: visible;
                display: block;
                position: center;
                top: 2px;
                }
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == "__main__":
    main()