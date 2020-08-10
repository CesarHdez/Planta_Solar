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
    
#new_data = data_generator(data, 5, 12, 6)

#data_p = tools.full_data_sun_hours(data, 'ENERGY')
#par = 'ENERGY'
#data_p = data_p[par]
#
#mask= 'day'
#groups = dict()
#
#for p in data_p.index:
#   if p.month not in groups:
#       groups[p.month] = []
#       groups[p.month].append(data_p.loc[p])
#   else:
#       groups[p.month].append(data_p.loc[p])
#
#labels, data_v = groups.keys(), groups.values()
#labels=["Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
#plt.boxplot(data_v)
#plt.xticks(range(1, len(labels) + 1), labels)
#plt.show()
#
######################################################
#
#
#par = 'ENERGY'
#data_p = data[par]
#mask= 'hour'
#groups = dict()
#
#for p in data_p.index:
#   if p.hour not in groups:
#       groups[p.hour] = []
#       groups[p.hour].append(data_p.loc[p])
#   else:
#       groups[p.hour].append(data_p.loc[p])
#
#labels, data_v = groups.keys(), groups.values()
##labels=["Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
#plt.boxplot(data_v)
#plt.xticks(range(1, len(labels) + 1), labels)
#plt.show()

########################################################
    
#data = tools.full_data_sun_hours(data, 'ENERGY')
#
#X = []
#for i in data.index:
#    X.append(i.hour)
#    
#Y = []
#for i in data.index:
#    Y.append(i.day)
#    
#Z = []
#for i in data["ENERGY"].values:
#    Z.append(i)
#
#
##X = np.array(X)
##Y = np.array(Y)
##Z = np.array(Z)
##
##fig = plt.figure()
##ax = plt.axes(projection='3d')
##ax.plot_trisurf(X, Y, Z, rstride=1, cstride=1,
##                cmap='viridis', edgecolor='none')
##ax.set_title('surface');
#
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#import pandas as pd
#import seaborn as sns
# 
## Get the data (csv file is hosted on the web)
##url = 'https://python-graph-gallery.com/wp-content/uploads/volcano.csv'
##data = pd.read_csv(url)
# 
## Transform it to a long format
##df=data.unstack().reset_index()
#df = pd.DataFrame()
##df.columns=["X","Y","Z"]
#df["X"]=Y
#df["Y"]=X
#df["Z"]=Z
## And transform the old column name in something numeric
##df['X']=pd.Categorical(df['X'])
##df['X']=df['X'].cat.codes
# 
## Make the plot
#fig = plt.figure(figsize=(10,7))
#ax = fig.gca(projection='3d')
#ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
#plt.show()
# 
## to Add a color bar which maps values to colors.
#surf=ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
#fig.colorbar( surf, shrink=0.5, aspect=15)
#plt.show()
# 
## Rotate it
#ax.view_init(30, 45)
#plt.show()
# 
## Other palette
#ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.jet, linewidth=0.3)
#plt.show()
#######################################################################


#data = pd.concat([data, tools.data_generator(data, 5, 12, 6)])
#data = pd.concat([data, tools.data_generator(data, 5, 7, 12)])
#data = pd.concat([data, tools.data_generator(data, 5, 1, 5)])
#data = pd.concat([data, tools.data_generator(data, 5, 7, 12)])

data = pd.concat([data, tools.data_generator(data, 5, 12, 8)])
data_m = pd.concat([data, tools.data_generator(data, 5, 12, 8)])

dataset = data

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






