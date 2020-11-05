
"""
Otros Graficos para el anális exploratorio. 
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import settings

data = pd.read_excel(settings.ex_data, sheet_name='data')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)

daily = data.resample('D').mean()
weekly = data.resample('W').mean()
monthly = data.resample('M').mean()
#---------------------------------------------------------
##Box por hora
#sns.set_style('ticks')
#fig, ax = plt.subplots()
#fig.set_size_inches(13, 7)
##sns.set_palette("bright")
#sns.factorplot(data.index.hour, "ENERGY", data=data, kind="box", ax=ax, palette='gnuplot')
#ax.set(ylabel='EAE (KW)', xlabel='Tiempo (Horas)')
#

#othees palettes CMRmap_r  gnuplot  YlOrBr
#--------------------------------------------------------

#---------------------------------------------------------
#Barra suma por mes
#monthly = data.resample('M').sum()
#monthly = monthly.loc['06-01-2019':'05-31-2020']
#
##monthly["m"]=monthly.index.month
##monthly = monthly.sort_values(["m"])
#
#sns.set_style('ticks')
#fig, ax = plt.subplots()
#fig.set_size_inches(12, 6)
#sns.barplot(x=monthly.index.strftime('%b'), y="ENERGY", data=monthly,ax=ax, color='c')
#ax.set(ylabel='EAE (KW)')
#sns.despine()

#---------------------------------------------------------
#Corr heat map
#data = data.rename(columns={'ENERGY': 'EAE'})
#plt.figure(figsize=(15, 10))
#heatmap = sns.heatmap(data.corr(), vmin=0, vmax=1, annot=True, annot_kws={"size":14}, 
#                      cmap='winter_r')
#heatmap.set_title('Mapa de Correlación', fontdict={'fontsize':18}, pad=12)

#othees palettes CMRmap_r  gnuplot  YlOrBr BrBG

#---------------------------------------------------------
##3d EAE
#data = data.loc['06-01-2019':'09-15-2019']
##data = tools.full_data_sun_hours(data, 'ENERGY')
#print(len(data))
#X = []
#for i in data.index:
#    X.append(i.hour)
#    
#Y = []
#temp_month = data.index[0].month
#count=0
#for i in data.index:
#    if i.month == temp_month:
#        Y.append(i.day + count)
#    else:
#        count = Y[-1]
#        Y.append(i.day + count)
#        temp_month = i.month
#        
#    
#Z = []
#for i in data["ENERGY"].values:
#    Z.append(i)
#
#
#X = np.array(X)
#Y = np.array(Y)
#Z = np.array(Z)
#
#df = pd.DataFrame()
##df.columns=["X","Y","Z"]
#df["X"]=Y
#df["Y"]=X
#df["Z"]=Z
#
## Make the plot
#fig = plt.figure(figsize=(10,7))
#ax = fig.gca(projection='3d')
#ax.set_xlabel('Horas')
#ax.set_ylabel('Días')
#ax.set_zlabel('EAE')
##ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
#ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.hot, linewidth=0)
#ax.set_facecolor('xkcd:gray')
##ax.view_init(90, 90)
##ax.view_init(15,180)
#plt.show()
#---------------------------------------------------------

#sns.set(rc={'figure.figsize':(10, 5)})
#data['ENERGY'].plot(linewidth=1)
#
#values = data.values
## specify columns to plot
##fields = ['EAE','IRRAD1','IRRAD2','IRRAD3','IRRAD4','IRRAD5','TEMP1','TEMP2','WS1','WS2', 'WANG']
#fields = ['ENERGY','IRRAD1','IRRAD2','IRRAD3','IRRAD4','IRRAD5','TEMP1','TEMP2','WS1','WS2', 'WANG']
#i = 1
## plot each column
#plt.figure(figsize=(25, 30))
#for j in fields:
#	plt.subplot(len(fields), 1, i)
#	plt.plot(data[j],label=j)
#	plt.title(j, y=0.5, loc='left')
#	i += 1
#plt.show()





#---------------------------------------------------------
#---------------------------------------------------------
#OThers Graphs
#---------------------------------------------------------
#Graficarlas todas en el mismo graph no buena
#fig,eje= plt.subplots()
#for i in ['WS1','IRRAD1','TEMP1','WANG', 'ENERGY']:
#    eje.plot(dataset[i],label=i)
#    eje.set_ylim(0,7000)
#    eje.legend()
#    eje.set_ylabel('Producción (GWh)')
#    eje.set_title('Tendencias en la Producción de electricidad')

##grafica varias juntas con el modo de una de area.
#fig,eje = plt.subplots()
#eje.plot(dataset['ENERGY'],color='black',label='Consumo')
#dataset[['IRRAD1','WS1']].plot.area(ax=eje,linewidth=0)
#eje.legend()
#eje.set_ylabel('Total Mensual (GWh)')
#

#groups = [1, 2, 3, 4]
#i = 1
## plot each column
#plt.figure()
#for group in groups:
#	plt.subplot(len(groups), 1, i)
#	plt.plot(values[:, group])
#	plt.title(dataset.columns[group], y=0.5, loc='left')
#	i += 1
#plt.show()

#groups = [0, 1, 2, 3]
#i = 1
## plot each column
#plt.figure()
#for group in groups:
#	plt.subplot(len(groups), 1, i)
#	plt.plot(values[:, group])
#	plt.title(dataset.columns[group], y=0.5, loc='right')
#	i += 1
#plt.show()
#
#
# fig,eje = plt.subplots()
# eje.plot(dataset['ENERGY'],color='black',label='Energy')
# eje2 = eje.twinx()
# eje2.plot(dataset['IRRAD1'],color='red',label='Solar Rad')
# eje.legend()
# eje.set_ylabel('Relacion Radiacion- Energía')

#Análisis de Correlación

#f, ax = plt.subplots(figsize=(10, 8))
#cor = data.astype(float).corr(method = 'pearson')
#print(cor)

#sbn.heatmap(cor, cmap='coolwarm',
#               square=True, ax=ax)
#sbn.heatmap(cor, mask=np.zeros_like(cor, dtype=np.bool), cmap=sbn.diverging_palette(200, 20, as_cmap=True),
#               square=True, ax=ax)
#sbn.heatmap(cor, mask=np.zeros_like(cor, dtype=np.bool), cmap=sbn.light_palette((100, 90, 60), input="husl"),
#               square=True, ax=ax)



#---------------------------------------------------------
