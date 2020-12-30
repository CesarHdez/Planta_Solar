#Settings

path = './data_files'

g_path = './graphics/'
m_path = './models/'
mk_path = './models_mk/'
analize_path = './to_analyze/'
ex_data = 'full_data.xlsx'
ex_data_2 = 'full_data_gen.xlsx'
conf_u_path = 'u_config.json'
conf_m_path = 'm_config.json'
conf_ml_path = 'lstm_m_config.json'
conf_path = 'lstm_config.json'
#conf_m_path = 'lstm_m_config.json'
conf_path_mpl = 'mpl_config.json'
conf_path_m_mpl = 'mpl_m_config.json'
exp_path = './experiments/'
app_models_path = './app_models/'
this_path = './'
tb_path = "logs\\fit\\"

#tb_path = "logs\\"

# g_path = '/content/solar_plant/graphics/'
# ex_data = '/content/solar_plant/full_data.xlsx'
# ex_data_2 = '/content/solar_plant/full_data_gen.xlsx'
# m_path = '/content/solar_plant/models/'
# mk_path = '/content/solar_plant/models_mk/'
#conf_u_path = '/content/solar_plant/u_config.json'
#conf_m_path = '/content/solar_plant/m_config.json'
# conf_path = '/content/solar_plant/lstm_config.json'
# conf_ml_path = '/content/solar_plant/lstm_m_config.json'
# conf_path_mpl = '/content/solar_plant/mpl_config.json'
# conf_path_m_mpl = '/content/solar_plant/mpl_m_config.json'
# exp_path = '/content/solar_plant/experiment/'
# this_path = '/content/solar_plant/'
# tb_path = this_path + "logs/"

headers_list=['DateTime' ,'WS1', 'WS2', 'ENERGY', 'IRRAD1',	'IRRAD2', 'IRRAD3',	'IRRAD4', 'IRRAD5',	'TEMP1', 'TEMP2', 'WANG']
headers_list_f=['DateTime' , 'ENERGY', 'WS1', 'WS2', 'IRRAD1',	'IRRAD2', 'IRRAD3',	'IRRAD4', 'IRRAD5',	'TEMP1', 'TEMP2', 'WANG']
headers_list_inv=['WS1', 'WS2', 'ENERGY', 'IRRAD1',	'IRRAD2', 'IRRAD3',	'IRRAD4', 'IRRAD5',	'TEMP1', 'TEMP2', 'WANG', 'DateTime']

cols2delete = ['WS2', 'IRRAD2', 'IRRAD3','IRRAD4', 'IRRAD5', 'TEMP2']


#-----------------------------------------------------
# %load_ext tensorboard
# %tensorboard --logdir /content/solar_plant/logs/

# !zip -r /content/expetiments.zip /content/solar_plant/experiment/
# !zip -r /content/logs.zip /content/solar_plant/logs/

# from google.colab import files
# files.download("/content/expetiments.zip")
# files.download("/content/logs.zip")

combinations = [(50, 'relu', 24, 'MSE', 'Adam'), (50, 'relu', 24, 'MSE', 'RMSprop'), (50, 'relu', 24, 'huber_loss', 'Adam'), (50, 'relu', 24, 'huber_loss', 'RMSprop'), (50, 'relu', 24, 'logcosh', 'Adam'), (50, 'relu', 24, 'logcosh', 'RMSprop'), (50, 'relu', 48, 'MSE', 'Adam'), (50, 'relu', 48, 'MSE', 'RMSprop'), (50, 'relu', 48, 'huber_loss', 'Adam'), (50, 'relu', 48, 'huber_loss', 'RMSprop'), (50, 'relu', 48, 'logcosh', 'Adam'), (50, 'relu', 48, 'logcosh', 'RMSprop'), (50, 'relu', 72, 'MSE', 'Adam'), (50, 'relu', 72, 'MSE', 'RMSprop'), (50, 'relu', 72, 'huber_loss', 'Adam'), (50, 'relu', 72, 'huber_loss', 'RMSprop'), (50, 'relu', 72, 'logcosh', 'Adam'), (50, 'relu', 72, 'logcosh', 'RMSprop'), (50, 'elu', 24, 'MSE', 'Adam'), (50, 'elu', 24, 'MSE', 'RMSprop'), (50, 'elu', 24, 'huber_loss', 'Adam'), (50, 'elu', 24, 'huber_loss', 'RMSprop'), (50, 'elu', 24, 'logcosh', 'Adam'), (50, 'elu', 24, 'logcosh', 'RMSprop'), (50, 'elu', 48, 'MSE', 'Adam'), (50, 'elu', 48, 'MSE', 'RMSprop'), (50, 'elu', 48, 'huber_loss', 'Adam'), (50, 'elu', 48, 'huber_loss', 'RMSprop'), (50, 'elu', 48, 'logcosh', 'Adam'), (50, 'elu', 48, 'logcosh', 'RMSprop'), (50, 'elu', 72, 'MSE', 'Adam'), (50, 'elu', 72, 'MSE', 'RMSprop'), (50, 'elu', 72, 'huber_loss', 'Adam'), (50, 'elu', 72, 'huber_loss', 'RMSprop'), (50, 'elu', 72, 'logcosh', 'Adam'), (50, 'elu', 72, 'logcosh', 'RMSprop'), (50, 'LeakyReLU', 24, 'MSE', 'Adam'), (50, 'LeakyReLU', 24, 'MSE', 'RMSprop'), (50, 'LeakyReLU', 24, 'huber_loss', 'Adam'), (50, 'LeakyReLU', 24, 'huber_loss', 'RMSprop'), (50, 'LeakyReLU', 24, 'logcosh', 'Adam'), (50, 'LeakyReLU', 24, 'logcosh', 'RMSprop'), (50, 'LeakyReLU', 48, 'MSE', 'Adam'), (50, 'LeakyReLU', 48, 'MSE', 'RMSprop'), (50, 'LeakyReLU', 48, 'huber_loss', 'Adam'), (50, 'LeakyReLU', 48, 'huber_loss', 'RMSprop'), (50, 'LeakyReLU', 48, 'logcosh', 'Adam'), (50, 'LeakyReLU', 48, 'logcosh', 'RMSprop'), (50, 'LeakyReLU', 72, 'MSE', 'Adam'), (50, 'LeakyReLU', 72, 'MSE', 'RMSprop'), (50, 'LeakyReLU', 72, 'huber_loss', 'Adam'), (50, 'LeakyReLU', 72, 'huber_loss', 'RMSprop'), (50, 'LeakyReLU', 72, 'logcosh', 'Adam'), (50, 'LeakyReLU', 72, 'logcosh', 'RMSprop'), (100, 'relu', 24, 'MSE', 'Adam'), (100, 'relu', 24, 'MSE', 'RMSprop'), (100, 'relu', 24, 'huber_loss', 'Adam'), (100, 'relu', 24, 'huber_loss', 'RMSprop'), (100, 'relu', 24, 'logcosh', 'Adam'), (100, 'relu', 24, 'logcosh', 'RMSprop'), (100, 'relu', 48, 'MSE', 'Adam'), (100, 'relu', 48, 'MSE', 'RMSprop'), (100, 'relu', 48, 'huber_loss', 'Adam'), (100, 'relu', 48, 'huber_loss', 'RMSprop'), (100, 'relu', 48, 'logcosh', 'Adam'), (100, 'relu', 48, 'logcosh', 'RMSprop'), (100, 'relu', 72, 'MSE', 'Adam'), (100, 'relu', 72, 'MSE', 'RMSprop'), (100, 'relu', 72, 'huber_loss', 'Adam'), (100, 'relu', 72, 'huber_loss', 'RMSprop'), (100, 'relu', 72, 'logcosh', 'Adam'), (100, 'relu', 72, 'logcosh', 'RMSprop'), (100, 'elu', 24, 'MSE', 'Adam'), (100, 'elu', 24, 'MSE', 'RMSprop'), (100, 'elu', 24, 'huber_loss', 'Adam'), (100, 'elu', 24, 'huber_loss', 'RMSprop'), (100, 'elu', 24, 'logcosh', 'Adam'), (100, 'elu', 24, 'logcosh', 'RMSprop'), (100, 'elu', 48, 'MSE', 'Adam'), (100, 'elu', 48, 'MSE', 'RMSprop'), (100, 'elu', 48, 'huber_loss', 'Adam'), (100, 'elu', 48, 'huber_loss', 'RMSprop'), (100, 'elu', 48, 'logcosh', 'Adam'), (100, 'elu', 48, 'logcosh', 'RMSprop'), (100, 'elu', 72, 'MSE', 'Adam'), (100, 'elu', 72, 'MSE', 'RMSprop'), (100, 'elu', 72, 'huber_loss', 'Adam'), (100, 'elu', 72, 'huber_loss', 'RMSprop'), (100, 'elu', 72, 'logcosh', 'Adam'), (100, 'elu', 72, 'logcosh', 'RMSprop'), (100, 'LeakyReLU', 24, 'MSE', 'Adam'), (100, 'LeakyReLU', 24, 'MSE', 'RMSprop'), (100, 'LeakyReLU', 24, 'huber_loss', 'Adam'), (100, 'LeakyReLU', 24, 'huber_loss', 'RMSprop'), (100, 'LeakyReLU', 24, 'logcosh', 'Adam'), (100, 'LeakyReLU', 24, 'logcosh', 'RMSprop'), (100, 'LeakyReLU', 48, 'MSE', 'Adam'), (100, 'LeakyReLU', 48, 'MSE', 'RMSprop'), (100, 'LeakyReLU', 48, 'huber_loss', 'Adam'), (100, 'LeakyReLU', 48, 'huber_loss', 'RMSprop'), (100, 'LeakyReLU', 48, 'logcosh', 'Adam'), (100, 'LeakyReLU', 48, 'logcosh', 'RMSprop'), (100, 'LeakyReLU', 72, 'MSE', 'Adam'), (100, 'LeakyReLU', 72, 'MSE', 'RMSprop'), (100, 'LeakyReLU', 72, 'huber_loss', 'Adam'), (100, 'LeakyReLU', 72, 'huber_loss', 'RMSprop'), (100, 'LeakyReLU', 72, 'logcosh', 'Adam'), (100, 'LeakyReLU', 72, 'logcosh', 'RMSprop'), (150, 'relu', 24, 'MSE', 'Adam'), (150, 'relu', 24, 'MSE', 'RMSprop'), (150, 'relu', 24, 'huber_loss', 'Adam'), (150, 'relu', 24, 'huber_loss', 'RMSprop'), (150, 'relu', 24, 'logcosh', 'Adam'), (150, 'relu', 24, 'logcosh', 'RMSprop'), (150, 'relu', 48, 'MSE', 'Adam'), (150, 'relu', 48, 'MSE', 'RMSprop'), (150, 'relu', 48, 'huber_loss', 'Adam'), (150, 'relu', 48, 'huber_loss', 'RMSprop'), (150, 'relu', 48, 'logcosh', 'Adam'), (150, 'relu', 48, 'logcosh', 'RMSprop'), (150, 'relu', 72, 'MSE', 'Adam'), (150, 'relu', 72, 'MSE', 'RMSprop'), (150, 'relu', 72, 'huber_loss', 'Adam'), (150, 'relu', 72, 'huber_loss', 'RMSprop'), (150, 'relu', 72, 'logcosh', 'Adam'), (150, 'relu', 72, 'logcosh', 'RMSprop'), (150, 'elu', 24, 'MSE', 'Adam'), (150, 'elu', 24, 'MSE', 'RMSprop'), (150, 'elu', 24, 'huber_loss', 'Adam'), (150, 'elu', 24, 'huber_loss', 'RMSprop'), (150, 'elu', 24, 'logcosh', 'Adam'), (150, 'elu', 24, 'logcosh', 'RMSprop'), (150, 'elu', 48, 'MSE', 'Adam'), (150, 'elu', 48, 'MSE', 'RMSprop'), (150, 'elu', 48, 'huber_loss', 'Adam'), (150, 'elu', 48, 'huber_loss', 'RMSprop'), (150, 'elu', 48, 'logcosh', 'Adam'), (150, 'elu', 48, 'logcosh', 'RMSprop'), (150, 'elu', 72, 'MSE', 'Adam'), (150, 'elu', 72, 'MSE', 'RMSprop'), (150, 'elu', 72, 'huber_loss', 'Adam'), (150, 'elu', 72, 'huber_loss', 'RMSprop'), (150, 'elu', 72, 'logcosh', 'Adam'), (150, 'elu', 72, 'logcosh', 'RMSprop'), (150, 'LeakyReLU', 24, 'MSE', 'Adam'), (150, 'LeakyReLU', 24, 'MSE', 'RMSprop'), (150, 'LeakyReLU', 24, 'huber_loss', 'Adam'), (150, 'LeakyReLU', 24, 'huber_loss', 'RMSprop'), (150, 'LeakyReLU', 24, 'logcosh', 'Adam'), (150, 'LeakyReLU', 24, 'logcosh', 'RMSprop'), (150, 'LeakyReLU', 48, 'MSE', 'Adam'), (150, 'LeakyReLU', 48, 'MSE', 'RMSprop'), (150, 'LeakyReLU', 48, 'huber_loss', 'Adam'), (150, 'LeakyReLU', 48, 'huber_loss', 'RMSprop'), (150, 'LeakyReLU', 48, 'logcosh', 'Adam'), (150, 'LeakyReLU', 48, 'logcosh', 'RMSprop'), (150, 'LeakyReLU', 72, 'MSE', 'Adam'), (150, 'LeakyReLU', 72, 'MSE', 'RMSprop'), (150, 'LeakyReLU', 72, 'huber_loss', 'Adam'), (150, 'LeakyReLU', 72, 'huber_loss', 'RMSprop'), (150, 'LeakyReLU', 72, 'logcosh', 'Adam'), (150, 'LeakyReLU', 72, 'logcosh', 'RMSprop')]
comb_50 = [(50, 'relu', 24, 'MSE', 'Adam'), (50, 'relu', 24, 'MSE', 'RMSprop'), (50, 'relu', 24, 'huber_loss', 'Adam'), (50, 'relu', 24, 'huber_loss', 'RMSprop'), (50, 'relu', 24, 'logcosh', 'Adam'), (50, 'relu', 24, 'logcosh', 'RMSprop'), (50, 'relu', 48, 'MSE', 'Adam'), (50, 'relu', 48, 'MSE', 'RMSprop'), (50, 'relu', 48, 'huber_loss', 'Adam'), (50, 'relu', 48, 'huber_loss', 'RMSprop'), (50, 'relu', 48, 'logcosh', 'Adam'), (50, 'relu', 48, 'logcosh', 'RMSprop'), (50, 'relu', 72, 'MSE', 'Adam'), (50, 'relu', 72, 'MSE', 'RMSprop'), (50, 'relu', 72, 'huber_loss', 'Adam'), (50, 'relu', 72, 'huber_loss', 'RMSprop'), (50, 'relu', 72, 'logcosh', 'Adam'), (50, 'relu', 72, 'logcosh', 'RMSprop'), (50, 'elu', 24, 'MSE', 'Adam'), (50, 'elu', 24, 'MSE', 'RMSprop'), (50, 'elu', 24, 'huber_loss', 'Adam'), (50, 'elu', 24, 'huber_loss', 'RMSprop'), (50, 'elu', 24, 'logcosh', 'Adam'), (50, 'elu', 24, 'logcosh', 'RMSprop'), (50, 'elu', 48, 'MSE', 'Adam'), (50, 'elu', 48, 'MSE', 'RMSprop'), (50, 'elu', 48, 'huber_loss', 'Adam'), (50, 'elu', 48, 'huber_loss', 'RMSprop'), (50, 'elu', 48, 'logcosh', 'Adam'), (50, 'elu', 48, 'logcosh', 'RMSprop'), (50, 'elu', 72, 'MSE', 'Adam'), (50, 'elu', 72, 'MSE', 'RMSprop'), (50, 'elu', 72, 'huber_loss', 'Adam'), (50, 'elu', 72, 'huber_loss', 'RMSprop'), (50, 'elu', 72, 'logcosh', 'Adam'), (50, 'elu', 72, 'logcosh', 'RMSprop'), (50, 'LeakyReLU', 24, 'MSE', 'Adam'), (50, 'LeakyReLU', 24, 'MSE', 'RMSprop'), (50, 'LeakyReLU', 24, 'huber_loss', 'Adam'), (50, 'LeakyReLU', 24, 'huber_loss', 'RMSprop'), (50, 'LeakyReLU', 24, 'logcosh', 'Adam'), (50, 'LeakyReLU', 24, 'logcosh', 'RMSprop'), (50, 'LeakyReLU', 48, 'MSE', 'Adam'), (50, 'LeakyReLU', 48, 'MSE', 'RMSprop'), (50, 'LeakyReLU', 48, 'huber_loss', 'Adam'), (50, 'LeakyReLU', 48, 'huber_loss', 'RMSprop'), (50, 'LeakyReLU', 48, 'logcosh', 'Adam'), (50, 'LeakyReLU', 48, 'logcosh', 'RMSprop'), (50, 'LeakyReLU', 72, 'MSE', 'Adam'), (50, 'LeakyReLU', 72, 'MSE', 'RMSprop'), (50, 'LeakyReLU', 72, 'huber_loss', 'Adam'), (50, 'LeakyReLU', 72, 'huber_loss', 'RMSprop'), (50, 'LeakyReLU', 72, 'logcosh', 'Adam'), (50, 'LeakyReLU', 72, 'logcosh', 'RMSprop')]