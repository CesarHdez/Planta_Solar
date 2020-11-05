from tensorflow.keras.models import load_model
import ml_tools
import settings
import tools
import os
import pandas as pd
import matplotlib.pyplot as plt

#model = load_model('Custom1_u.h5')
#model.summary()


#settings
save = True
save_path = './saved/'

model_type = '_m'



list_perf = []
list_relat = []



with os.scandir(settings.analize_path) as folders:
    folders = [folders.name for folders in folders if folders.is_dir()]  
    
for folder in folders:
    with os.scandir(settings.analize_path +'/' +folder ) as files:
        files = [files.name for files in files if files.is_file() and files.name.endswith(model_type +'.pkl')]  
    m_perf = ml_tools.load_perf(settings.analize_path +'/' + folder +'/' +files[0])
    list_perf.append(m_perf)
    with os.scandir(settings.analize_path +'/' +folder ) as files:
        files = [files.name for files in files if files.is_file() and files.name.endswith(model_type +'_fc_dt.pkl')]
    m_relat = ml_tools.load_perf(settings.analize_path +'/' + folder +'/' +files[0])
    list_relat.append(m_relat)

#---------------------------------------------------------------
models_name = ["Modelo 1", "Modelo 2"]
models_df= pd.DataFrame()
for i in range(len(list_relat)):
    if i == 0:
        models_df = list_relat[i]
        #models_df = models_df.rename(columns={'ENERGY': 'Real','forecast': 'Model_'+str(i)})
        models_df = models_df.rename(columns={'ENERGY': 'Real','forecast': models_name[i]})
    else:
        temp_list = list(list_relat[i]['forecast'].tolist())
        models_df[models_name[i]]= temp_list
        #models_df['Model_'+str(i)]= temp_list
#---------------------------------------------------------------
models_df= models_df[500:700]
names_list = list(models_df.columns)
fig,eje= plt.subplots(figsize=(10,5))
title = 'Comparación de pronósticos de los Modelos'
for i in names_list:
    eje.plot(models_df[i],label=i)
    eje.set_ylim(-150,11000)
    eje.legend()
    eje.set_ylabel('Producción (MWh)')
    eje.set_title(title)
if save:
    fig.savefig('./to_analyze/'+title+'.png')
#---------------------------------------------------------------
r_list=[]
rr_list=[]
for relat in list_relat:
    cor = relat.astype(float).corr(method = 'pearson').iloc[1][0]
    r_list.append(cor)
    rr = ml_tools.det_coef(relat["ENERGY"].values, relat["forecast"].values)
    rr_list.append(rr)

#y_true = np.array(relat["ENERGY"])
#y_pred = np.array(relat["forecast"])
#MSE = np.square(np.subtract(y_true, y_pred)).mean()
#RMSE = sqrt(MSE)
#print("MSE: ", MSE)
#print("RMSE: ", RMSE)

#print(ml_tools.get_model_stats(m_perf.history))


#---------------------------------------------------------------
metric = 'loss'
fig,eje= plt.subplots(figsize=(10.3,5))
legend_list=[]
for i in range(len(list_perf)):
    m_perf = list_perf[i] 
    if metric in m_perf:
        eje.plot(m_perf[metric])
        val_m = 'val_'+metric
        if val_m in m_perf:
            eje.plot(m_perf[val_m])
        title = 'Comparación de Modelos '+ metric
        title = 'Pérdida en el desempeño del entrenamiento de los Modelos'
        eje.set_ylabel('Pérdida')
        #eje.set_ylabel(metric)
        eje.set_xlabel('época')
        eje.set_title(title)
        eje.set_ylim(-0.001,0.12)
#        legend_list.append('Model '+ str(i) +' Train')
#        legend_list.append('Model '+ str(i) +' Test')
        legend_list.append(models_name[i] +' Entrenamiento')
        legend_list.append(models_name[i] +' Prueba')
        eje.legend(legend_list, loc='upper right')
        eje.grid()
#        if save:
#            eje.savefig(settings.g_path+title+'.png')
#    	eje.grid()
#    	eje.show()
    else:
    	print('Metric no calculated')
if save:
    fig.savefig('./to_analyze/'+title+'.png')
#-----------------------------------------------------------------
#metrics = ["mse", "mae", "mape"]
#metric_list=[]
#for i in range(len(list_perf)):
#    m_perf = list_perf[i]
#    temp = []
#    for k in metrics:     
#        temp.append(m_perf[k][-1])
#    metric_list.append(temp)
    
metrics = ["mse", "mae", "mape", "root_mean_squared_error"]
metric_list=[]
for k in metrics:
    temp = []
    for m_perf in list_perf:
        temp.append(m_perf[k][-1])
    metric_list.append(temp)

#-----------------------------------------------------------------
models_stats= pd.DataFrame()
columns_l=['name','corr', 'det'] + metrics + ['exprmt']
models_stats['name']=models_name
models_stats['corr']=r_list
models_stats['det']=rr_list
models_stats['exprmt']=folders
for i in range(len(metrics)):
    if metrics[i] == "root_mean_squared_error":
        models_stats['rmse']=metric_list[i]
    else:
        models_stats[metrics[i]]=metric_list[i]
models_stats = models_stats.set_index('exprmt')
print(models_stats)
if save:
    models_stats.to_excel('./to_analyze/models_stats.xlsx', sheet_name='stats')
    






