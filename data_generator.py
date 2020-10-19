import pandas as pd
import settings
import random
import datetime

import matplotlib.pyplot as plt
import numpy as np
import tools
#import numpy as np


data = pd.read_excel(settings.ex_data, sheet_name='data')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)

data_new = pd.read_excel('full_data.xlsx', sheet_name='data')
data_new['DateTime'] = pd.to_datetime(data_new['DateTime'])
data_new.set_index('DateTime', inplace=True)

#data = data[:-140]
data = data.astype(float)

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
    


def shufle_data_year(data, group):
    num_m =[6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5]
    #num_m =[10, 11, 12]
    list_df = []
    for  i in num_m:
        data_m = month_selector(data, i)
        list_d = day_spliter(data_m)
        list_g = month_groups_random(list_d, group)
        random.shuffle(list_g)
        m_df = reconst_df(list_g)
        list_df.append(m_df)
    shuffled_df = reconst_df(list_df)
    return shuffled_df

def data_generator_year(data, group, last_date):
    sh_df = shufle_data_year(data, group)
    #date = list(data.index)[-1]
    date = last_date
    date_time_col = []
    for i in range(len(sh_df)):
        date_time_col.append(date + datetime.timedelta(hours= 1))
        date = date_time_col[-1]
    
    sh_df["DateTime"]= date_time_col
    sh_df.set_index("DateTime", inplace = True)
    return sh_df


data_new = data_new[:'05-31-2020']
##print(len (data_c))
data_new_m = pd.concat([data_new, tools.data_generator_year(data_new, 5, list(data_new.index)[-1])])

#data = pd.concat([data, tools.data_generator(data, 5, 10, 12)])
#data = pd.concat([data, tools.data_generator(data, 5, 1, 5)])
#data_m = pd.concat([data, tools.data_generator_year(data, 5, list(data.index)[-1])])


#data = pd.concat([data, tools.data_generator(data, 5, 12, 6)])
#data = pd.concat([data, tools.data_generator(data, 5, 7, 12)])
#data = pd.concat([data, tools.data_generator(data, 5, 1, 5)])
#data = pd.concat([data, tools.data_generator(data, 5, 7, 12)])

data = pd.concat([data, tools.data_generator(data, 5, 12, 8)])
data_c = data[:-24]
#print(len (data_c))
data_m = pd.concat([data_c, tools.data_generator_year(data_c, 5, list(data_c.index)[-1])])
#print(len(data_m), list(data_c.index)[-1])
#data_m = pd.concat([data_m, tools.data_generator_year(data_c, 5, list(data_m.index)[-1])])
##print(len (data_m), list(data_m.index)[-1])
#data_m = pd.concat([data_m, tools.data_generator_year(data_c, 5, list(data_m.index)[-1])])
##print(len (data_m), list(data_m.index)[-1])

data_m.to_excel('full_data_gen.xlsx', sheet_name='data')

#enero= data_m.loc['01-01-2020':'01-31-2020']
#enero.to_excel('enero.xlsx', sheet_name='data')

dataset = data_m

values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3, 4]
i = 1
# plot each column
plt.figure()
for group in groups:
	plt.subplot(len(groups), 1, i)
	plt.plot(values[:, group])
	plt.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
plt.show()






