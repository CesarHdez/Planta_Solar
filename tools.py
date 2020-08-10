from openpyxl import Workbook,load_workbook,cell
import numpy as np
import pandas as pd
import datetime
from os import scandir
import settings
import random

def ls2(path):
    return [obj.name for obj in scandir(path) if obj.is_file()]


def fix_excel(filename):
	headers_list= settings.headers_list[1:]
	raw_data = pd.read_excel(filename, 'Data')
	raw_data['DateTime'] = raw_data['Date'] +' '+raw_data['Time']
	raw_data.drop(['Date', 'Time'], axis=1, inplace=True)
	col_names =list(raw_data.columns)
	raw_data = raw_data.reindex(columns =['DateTime'] + col_names[:-1])
	raw_data.columns = settings.headers_list
	raw_data = format_dataframe(raw_data, 'DateTime')
	raw_data = raw_data[:-1]
	return raw_data


def fill_na_col_daym(data, col):
	null_list = list(data[col].isnull().values)
	for i in range(len(null_list)):
		if null_list[i] == True:
			if data.index.get_level_values(0)[i].month == data.index.get_level_values(0)[1].month:
				fecha = data.index.get_level_values(0)[i]
				#print(fecha, col)
				if fecha.day > 7:
					aux_l = []
					for j in range(1,8):
						if (fecha-datetime.timedelta(days=j)) in data.index:
							aux_l.append(data.loc[(fecha-datetime.timedelta(days=j))][col])
						else:
							aux_l.append(0)
					data[col][i] = np.array(aux_l).mean()
				else:
					aux_l = []
					for j in range(1,8):
						if (fecha+datetime.timedelta(days=j)) in data.index:
							aux_l.append(data.loc[(fecha+datetime.timedelta(days=j))][col])
						else:
							aux_l.append(0)
					data[col][i] = np.array(aux_l).mean()
			else:
				pass
				#data.drop(i, axis = 0)
	return data


def fill_na_all(data):
	par_list = ['IRRAD1', 'IRRAD2', 'IRRAD3', 'IRRAD4', 'IRRAD5', 'TEMP1', 'TEMP2']
	for i in par_list:
		data = fill_na_col_daym(data, i)
	data['WS1'].fillna(method = 'backfill', inplace = True)
	data['WS2'].fillna(method = 'backfill', inplace = True)
	data['WANG'].fillna(method = 'backfill', inplace = True)
	par_list = ['IRRAD1', 'IRRAD2', 'IRRAD3', 'IRRAD4', 'IRRAD5', 'TEMP1', 'TEMP2']
	for i in par_list:
		data = fill_na_col_daym(data, i)
	return data


def last_day_of_month(any_day):
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)  # this will never fail
    return next_month - datetime.timedelta(days=next_month.day)


def resample_to_hour(data):
	i = 0
	fecha_init = data.index.get_level_values(0)[0]
	fecha = fecha_init
	last_href = last_day_of_month(fecha) + datetime.timedelta(hours=23)
	resp_data = pd.DataFrame(columns= settings.headers_list_f, index= range(24*last_href.day))
	while fecha <= last_href:
		if fecha != fecha_init:
			resp_data['ENERGY'][i-1]= data.loc[fecha]['ENERGY']
		resp_data['DateTime'][i] = fecha
		for j in list(data.columns)[1:]:
			resp_data[j][i]= np.array(data.loc[str(fecha)[:13]][j]).mean()
		i = i + 1
		fecha = fecha + datetime.timedelta(hours=1)
	resp_data['ENERGY'][i-1]= 0 
	resp_data['DateTime'] = pd.to_datetime(resp_data['DateTime'])
	resp_data.set_index('DateTime', inplace=True)
	if resp_data['ENERGY'].isnull().values.any():
		resp_data = fill_na_col_daym(resp_data, 'ENERGY')
	return resp_data
    

def add_day(fecha):
    return (fecha + datetime.timedelta(days=1))


def add_data_month(full_data, month_file_name):    
	data_fx = fix_excel(month_file_name)
	data_fx = fill_na_all(data_fx)
	data_fx = resample_to_hour(data_fx)
	if full_data.empty:
		full_data = data_fx
	else:
		full_data = pd.concat([full_data, data_fx])
	return full_data


def format_dataframe(dataframe, index):
    dataframe[index] = pd.to_datetime(dataframe[index])
    dataframe.set_index(index, inplace=True)
    dataframe = dataframe.sort_values([index])
    cols = dataframe.columns.tolist()
    cols = [cols[2]] + cols[:2] + cols[3:]
    dataframe = dataframe[cols]
    return dataframe


def negative_to_zero(full_data, par):
	#full_data[par] = full_data[par].clip(lower = 0)
	full_data.loc[full_data[par] < 0, par] = 0
	return full_data


def negative_to_positive(full_data, par):
	full_data.loc[full_data[par] < 0, par] = full_data[par] * -1
	return full_data


def neg_irrad_2_zero(full_data):
	irrad_name = ['IRRAD1','IRRAD2','IRRAD3','IRRAD4','IRRAD5']
	for col in irrad_name:
		full_data = negative_to_zero(full_data, col)
	return full_data


#elimina del data frame las filas en las que no se genera energía (horas de sol)
def full_data_sun_hours(full_data, par):
	full_data.drop(full_data[full_data[par] == 0].index, inplace = True)
	return full_data


def delete_cols(full_data, cols2delete):
	full_data = full_data.drop(cols2delete, axis=1)
	return full_data


def change_outliers_values(full_data, par):
	q1 = full_data[par].quantile(0.25)
	q3 = full_data[par].quantile(0.75)
	iqr = q3-q1 #Interquartile range
	#print(q1,q3, iqr)
	#fence_low  = q1-1.5*iqr
	#fence_high = q3+1.5*iqr
	#print(fence_high)
    #full_data.loc[full_data[par] > 10000, par] = 8000
	quant = 0.99
	#threshold = full_data[par].quantile(quant)
	threshold = 10000
	#print("threhold: " ,threshold)
	full_data.loc[full_data[par] > threshold, par] = full_data[par].quantile(quant)
	return full_data

def data_split(array, percent):
	limit = int(len(array) * percent / 100)
	train, test = array[:limit,:], array[limit:,:]
	return train, test, limit

#########################################################################
#Data Generator
#########################################################################

def month_selector(data, month):
    return data[data.index.month == month]

def day_spliter(data_m):
    last_day = list(data_m.index)[-1].day
    list_of_days = []
    for i in range(1,last_day+1):
        list_of_days.append(data_m[data_m.index.day == i])
    return list_of_days

def month_groups_random(list_d, group):
    res = len(list_d) % group
    days_groups =[]
    
    for i in range(0, len(list_d), group):
        df_aux = pd.DataFrame()
        if i + (group - 1) > (len(list_d)-1):
            for j in range(i, i + (res)):
                if df_aux.empty:
                    df_aux = list_d[j]
                else:
                    df_aux = pd.concat([df_aux, list_d[j]])
            days_groups.append(df_aux)
        else:
            for j in range(i, i + group):
                if df_aux.empty:
                    df_aux = list_d[j]
                else:
                    df_aux = pd.concat([df_aux, list_d[j]])
            days_groups.append(df_aux)
    return days_groups


def reconst_df(list_g):
    df_aux = pd.DataFrame()
    for i in range(len(list_g)):
        if df_aux.empty:
            df_aux = list_g[i]
        else:
            df_aux = pd.concat([df_aux, list_g[i]])
    return df_aux

#data_m = month_selector(data, 6)
#list_d = day_spliter(data_m)
#list_g = month_groups_random(list_d, 5)
#random.shuffle(list_g)
#m_df = reconst_df(list_g)


def shufle_data(data, group, init_m, end_m):
    list_df = []
    if init_m > end_m:
        for  i in range(init_m, end_m - 1, -1):
            data_m = month_selector(data, i)
            list_d = day_spliter(data_m)
            list_g = month_groups_random(list_d, group)
            random.shuffle(list_g)
            m_df = reconst_df(list_g)
            list_df.append(m_df)
    else:
        for  i in range(init_m, end_m + 1):
            data_m = month_selector(data, i)
            list_d = day_spliter(data_m)
            list_g = month_groups_random(list_d, group)
            random.shuffle(list_g)
            m_df = reconst_df(list_g)
            list_df.append(m_df)
    shuffled_df = reconst_df(list_df)
    return shuffled_df

def data_generator(data, group, init_m, end_m):
    sh_df = shufle_data(data, group, init_m, end_m)
    date = list(data.index)[-1]
    date_time_col = []
    for i in range(len(sh_df)):
        date_time_col.append(date + datetime.timedelta(hours= 1))
        date = date_time_col[-1]
    
    sh_df["DateTime"]= date_time_col
    sh_df.set_index("DateTime", inplace = True)
    return sh_df