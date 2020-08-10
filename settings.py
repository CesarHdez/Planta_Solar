#Settings

path = './data_files'

g_path = './graphics/'
m_path = './models/'
ex_data = 'full_data.xlsx'
conf_u_path = 'u_config.json'
conf_m_path = 'm_config.json'
conf_ml_path = 'lstm_m_config.json'
conf_path = 'lstm_config.json'
#conf_m_path = 'lstm_m_config.json'
conf_path_mpl = 'mpl_config.json'
conf_path_m_mpl = 'mpl_m_config.json'
exp_path = './experiments/'
this_path = './'
tb_path = "logs\\fit\\"

#tb_path = "logs\\"

# g_path = '/content/solar_plant/graphics/'
# ex_data = '/content/solar_plant/full_data.xlsx'
# m_path = '/content/solar_plant/models/'
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