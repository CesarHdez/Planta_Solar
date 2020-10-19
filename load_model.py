from tensorflow.keras.models import load_model
import ml_tools
import settings
import tools
import os
import pandas as pd
import matplotlib.pyplot as plt

#model = load_model('Custom1_u.h5')
#model.summary()


model_type = '_u'

#dirs_name = tools.ls2_dir(settings.analize_path)

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
models_name = ["LSTM", "GRU"]
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
models_df= models_df[500:-137]
names_list = list(models_df.columns)
fig,eje= plt.subplots()
for i in names_list:
    eje.plot(models_df[i],label=i)
    eje.set_ylim(-100,11000)
    eje.legend()
    eje.set_ylabel('Producción (MWh)')
    eje.set_title('Tendencias en la Producción de electricidad')

#for folder in dirs_name:
#    print(folder)


#m_perf = ml_tools.load_perf('Custom2_m.pkl')
#stats = ml_tools.get_model_stats(m_perf)
#print(stats)
    
    
#with os.scandir(settings.analize_path +'/'+folder) as ficheros:
#        ficheros = [fichero.name for fichero in ficheros if fichero.is_file() and fichero.name.endswith(model_type +'pkl')]