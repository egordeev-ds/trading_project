#imports
from callables import prepare_dataset
from callables import process_dataset
from callables import model_build

#params
filepath_input = '/Users/cotangentofzero/Data_science/Students/done/Ivan/prod/data/raw_data/'
model_name = "/Users/cotangentofzero/Data_science/Students/done/Ivan/prod/artifacts/trading_model_2"
shift = -9
anomaly_crtiretion = 1.0050

#callables
df = prepare_dataset(filepath_input)
df_event, df_proc = process_dataset(df, shift, anomaly_crtiretion)
model_build(df_proc, model_name)

#success
print('Success!')