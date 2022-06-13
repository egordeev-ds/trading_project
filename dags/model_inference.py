#imports
import yaml
from callables import prepare_dataset
from callables import process_dataset
from callables import model_inference
from utils import plot_candletick_anomaly
from callables import save_result
import warnings
warnings.filterwarnings('ignore')

#params
with open("params.yml", 'r') as file:
    params = yaml.safe_load(file)

root_path = params['root_path']
filepath_input = root_path+params['filepath_input']
filepath_output = root_path+params['filepath_output']
model_name = root_path+params['model_name']

shift = params['shift']
anomaly_crtiretion = params['anomaly_crtiretion']
cb_params = params['cb_params']
cutoff = params['cutoff']

#callables
df = prepare_dataset(filepath_input)
df_event, df_proc = process_dataset(df,shift, anomaly_crtiretion)
df_event = model_inference(df_proc,df_event,cutoff,model_name)
plot_candletick_anomaly(df_event,filepath_output, left_b = 5000, right_b = 10000)
save_result(df,filepath_output)

#success
print('Success!')