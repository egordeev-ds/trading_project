#imports
from callables import prepare_dataset
from callables import process_dataset
from callables import model_inference
from callables import plot_candletick_anomaly
from callables import save_result
import warnings
warnings.filterwarnings('ignore')

#params
filepath_input = '/Users/cotangentofzero/Data_science/Students/done/Ivan/prod/data/raw_data/'
filepath_output = '/Users/cotangentofzero/Data_science/Students/done/Ivan/prod/data/'
model_name = "/Users/cotangentofzero/Data_science/Students/done/Ivan/prod/artifacts/trading_model_2"
shift = -9
anomaly_crtiretion = 1.0050

#callables
df = prepare_dataset(filepath_input)
df_event, df_proc = process_dataset(df,shift, anomaly_crtiretion)
df_event = model_inference(df_proc,df_event,model_name)
plot_candletick_anomaly(df_event,filepath_output, left_b = 5000, right_b = 10000)
save_result(df,filepath_output)

#success
print('Success!')