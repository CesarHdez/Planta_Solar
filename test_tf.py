#import tensorflow as tf
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sbn

import settings
import tools

#Obtener los nombres de los archivos para procesar
files_name = tools.ls2(settings.path)

full_data = pd.DataFrame(columns= settings.headers_list)

for i in files_name:
    full_data =tools.add_data_month(full_data, settings.path +'/' + i)

full_data = tools.format_dataframe(full_data, settings.headers_list[0])

diary = full_data.resample('D').mean()
weekly = full_data.resample('W').mean()
monthly = full_data.resample('M').mean()

sbn.set(rc={'figure.figsize':(15, 5)})

full_data['TEMP1'].plot(linewidth=1)


#ts['MA'] = ts['MA'].fillna(method='backfill')