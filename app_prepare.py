import pickle
import app_tools
import settings

list_config, list_model = app_tools.get_lib_models()

with open(settings.app_src_path+'list_config.pkl', 'wb') as f:
    pickle.dump(list_config, f)
    

with open(settings.app_src_path+'list_model.pkl', 'ab') as f:
    _ = [pickle.dump(model, f) for model in list_model]