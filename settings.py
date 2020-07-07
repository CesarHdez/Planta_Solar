#Settings

path = './data_files'
g_path = './graphics/'
m_path = './models/'

headers_list=['DateTime' ,'WS1', 'WS2', 'ENERGY', 'IRRAD1',	'IRRAD2', 'IRRAD3',	'IRRAD4', 'IRRAD5',	'TEMP1', 'TEMP2', 'WANG']
headers_list_f=['DateTime' , 'ENERGY', 'WS1', 'WS2', 'IRRAD1',	'IRRAD2', 'IRRAD3',	'IRRAD4', 'IRRAD5',	'TEMP1', 'TEMP2', 'WANG']
headers_list_inv=['WS1', 'WS2', 'ENERGY', 'IRRAD1',	'IRRAD2', 'IRRAD3',	'IRRAD4', 'IRRAD5',	'TEMP1', 'TEMP2', 'WANG', 'DateTime']

cols2delete = ['WS2', 'IRRAD2', 'IRRAD3','IRRAD4', 'IRRAD5', 'TEMP2']

#-----------------------------------------------------

#LSTM settings
# ml_conf = {
# 	y_var : 'ENERGY',
	# batch_size : 250,
	# epoch : 10,
	# lstm1 : 120,
	# lstm2 : 60,
	# lstm3 : 40,
	# past_hist : 120,
	# future_target : 0,
	# split_p : 90,
	# metrics : ['mse', 'mae', 'mape', 'cosine']
# }
# y_var = 'ENERGY'
# batch_size = 250
# epoch = 10
# lstm1 = 120
# lstm2 = 60
# lstm3 = 40
# past_hist = 120
# future_target = 0
# split_p = 90
# metrics = ['mse', 'mae', 'mape', 'cosine']

