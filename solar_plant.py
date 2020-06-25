import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


import settings
import tools

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def data_prep():	
	files_name = tools.ls2(settings.path)
	full_data= pd.DataFrame()
	for i in files_name:
	    full_data =tools.add_data_month(full_data, settings.path +'/' + i)
	full_data = full_data.sort_index(axis=0)
	full_data = tools.neg_irrad_2_zero(full_data)
	full_data = tools.negative_to_positive(full_data, 'ENERGY')
	full_data = tools.change_outliers_values(full_data, 'ENERGY')
	full_data = tools.full_data_sun_hours(full_data, 'ENERGY')
	full_data = tools.delete_cols(full_data, settings.cols2delete)
	full_data = full_data.astype(float)
	return full_data

full_data = data_prep()
daily = full_data.resample('D').mean()


print(full_data)