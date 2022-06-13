#imports
import yaml
from callables import prepare_dataset
from callables import process_dataset
from callables import model_train

#params
with open("params.yml", 'r') as file:
    params = yaml.safe_load(file)

root_path = params['root_path']
filepath_input = root_path+params['filepath_input']
model_name = root_path+params['model_name']

shift = params['shift']
anomaly_crtiretion = params['anomaly_crtiretion']
cb_params = params['cb_params']

#callables
df = prepare_dataset(filepath_input)
df_event, df_proc = process_dataset(df, shift, anomaly_crtiretion)
model_train(df_proc, cb_params, model_name)

#success
print('Success!')