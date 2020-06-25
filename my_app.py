import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt
import os
os.environ['TZ'] = 'UTC'


st.title('Copiapo Solar Plant')

#load_state = st.text('Loading data...')
#@st.cahe
data = pd.read_excel('full_data.xlsx', sheet_name='data')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)
data = data.astype(float)

#load_state = st.text('Loading data..done!')

st.header('Data Fixed')
st.dataframe(data)

st.subheader('Registers by hours')
daily = data.resample('D').mean()
#daily = daily.index.dt.tz_localize('UTC')
# auxl=list(daily.index)
# daily['Date'] = auxl
# daily.set_index('Date')

st.text('Energy')
start_date = st.sidebar.date_input('Start Date', data.index.get_level_values(0)[0])
end_date = st.sidebar.date_input('End Date', data.index.get_level_values(0)[-1])
st.line_chart(data[start_date:end_date]['ENERGY'])
st.line_chart(data[start_date:end_date]['WS1'])
st.line_chart(data[start_date:end_date]['IRRAD1'])
st.line_chart(data[start_date:end_date]['TEMP1'])
st.line_chart(data[start_date:end_date]['WANG'])
#print(daily.index)
