#imports
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score,roc_auc_score
from utils import plot_roc_curve, find_max_fscore, plot_confusion_matrix, plot_feature_importnaces

def prepare_dataset(filepath_input):

    #parsing json

    files=os.listdir(filepath_input)
    kline_js=[]
    for name in files:
        f=open(filepath_input+name)
        js=json.load(f)
        for i in range(len(js)):
            kline_js.append(js[i])

    E=[]
    t_start=[]
    t_end=[]
    opens=[]
    closes=[]
    v=[]
    n=[]
    h=[]
    l=[]
    x=[]
    q=[]
    V=[]
    Q=[]

    for js_i in kline_js:
        
        if "stream" in js_i["r"] and js_i["r"]["stream"] == "btcusdt@kline_1m":
            
            try:
                E.append(js_i['r']['data']['E'])
                t_start.append(js_i['r']['data']['k']['t'])
                t_end.append(js_i['r']['data']['k']['T'])
                opens.append(float(js_i['r']['data']['k']['o']))
                closes.append(float(js_i['r']['data']['k']['c']))
                v.append(float(js_i['r']['data']['k']['v']))
                n.append(js_i['r']['data']['k']['n'])
                x.append(js_i['r']['data']['k']['x'])
                q.append(float(js_i['r']['data']['k']['q']))
                V.append(float(js_i['r']['data']['k']['V']))
                Q.append(float(js_i['r']['data']['k']['Q']))
                h.append(float(js_i['r']['data']['k']['h']))
                l.append(float(js_i['r']['data']['k']['l']))  
            except:
                print(js_i)

    d = {
        'event_time': E,
        't_start':t_start, 
        't_end':t_end, 
        'open_price':opens, 
        'close_price':closes,
        'low_price':l, 
        'high_price':h,
        'base_volume':v,
        'quote_volume':q, 
        'buy_base':V, 
        'buy_quote':Q, 
        'n_trades':n, 
        'is_closed':x
        }

    df = pd.DataFrame(data=d)
    df['vol_per_trade'] = df['quote_volume'] / df['n_trades']

    #prepare features

    df['event_time'] = pd.to_datetime(
        df['event_time'].apply(
            lambda x: datetime.fromtimestamp(x/1000.0)
            ))

    df['t_start'] = pd.to_datetime(
        df['t_start'].apply(
            lambda x: datetime.fromtimestamp(x/1000.0)
            ))

    df['t_end'] = pd.to_datetime(
        df['t_end'].apply(
            lambda x: datetime.fromtimestamp(x/1000.0)
            ))

    df['close_delta'] = df['close_price'].shift(-1) - df['close_price']
    df['trades_delta'] = df['n_trades'].shift(-1) - df['n_trades']
    df['quote_volume_delta'] = df['quote_volume'].shift(-1) - df['quote_volume']
    df['base_volume_delta'] = df['base_volume'].shift(-1) - df['base_volume']
    df['buy_quote_delta'] = df['buy_quote'].shift(-1) - df['buy_quote']
    df['buy_base_delta'] = df['buy_base'].shift(-1) - df['buy_base']
    df['vol_per_trade_delta'] = df['vol_per_trade'].shift(-1) - df['vol_per_trade']

    df = df.drop_duplicates('event_time', keep = 'first')
    df = df.sort_values(by='event_time').reset_index(drop = True)
    df = df.dropna()

    return df

def process_dataset(df, shift_backwards, shift_forward, anomaly_crtiretion):

    import warnings
    warnings.filterwarnings('ignore')

    #change type
    df.t_start = pd.to_datetime(df.t_start)
    df.t_end = pd.to_datetime(df.t_end)

    #correction
    df['trades_delta'].loc[df['is_closed'] == True] = (df[df['is_closed'] == True]['n_trades'].values) + (df[df['is_closed'] == True]['trades_delta'].values)
    df['quote_volume_delta'].loc[df['is_closed'] == True] = (df[df['is_closed'] == True]['quote_volume'].values) + (df[df['is_closed'] == True]['quote_volume_delta'].values)
    df['base_volume_delta'].loc[df['is_closed'] == True] = (df[df['is_closed'] == True]['base_volume'].values) + (df[df['is_closed'] == True]['base_volume_delta'].values)
    df['buy_quote_delta'].loc[df['is_closed'] == True] = (df[df['is_closed'] == True]['buy_quote'].values) + (df[df['is_closed'] == True]['buy_quote_delta'].values)
    df['vol_per_trade_delta'].loc[df['is_closed'] == True] = (df[df['is_closed'] == True]['vol_per_trade'].values) + (df[df['is_closed'] == True]['vol_per_trade_delta'].values)

    #аномалия на начало периода
    df['anomaly_t_start'] = np.where(df.close_price > df.open_price,
                                    df.high_price/df.open_price,
                                    df.open_price/df.low_price)

    #аномалия на конец периода
    df['anomaly_t_end'] = df['anomaly_t_start'].shift(shift_forward)

    #target - на конец периода (спустя 250мс после t_start)
    df['target'] = df['anomaly_t_end'] > anomaly_crtiretion

    #округление
    #df = df.round(2)

    #select_состояние
    df_event = df[['t_start','open_price','close_price','low_price','high_price','n_trades','base_volume','quote_volume','buy_base','target']]

    #select deltas
    df_period = df[['t_start','t_end','close_delta','trades_delta','quote_volume_delta','buy_quote_delta','vol_per_trade_delta','target']]

    #features
    df_model = df_period.drop(['t_start', 't_end'], axis = 1)

    data_temp = pd.DataFrame()
    for col in df_model.drop('target', axis = 1).columns:
        for shift_bakcwards in range(shift_backwards+1):
            data_temp[f'{col}_{shift_bakcwards}'] = df_model[col].shift(shift_bakcwards)

    df_model = pd.concat([data_temp, df_model.target], axis = 1)
    df_model = df_model.dropna().reset_index(drop = True)

    return df_event, df_period, df_model
    
def model_train(data, cb_params, model_name):

    #split
    train, test = train_test_split(data, test_size = 0.2, random_state = 42, shuffle = False)

    x_train = train.drop('target', axis = 1)
    y_train = train.target.astype(int)

    x_test = test.drop('target', axis = 1)
    y_test = test.target.astype(int)

    #fit best model
    model = CatBoostClassifier(**cb_params)
    model.fit(x_train, y_train)

    # predict train probabilities
    y_train_pred_proba = model.predict_proba(x_train)
    y_train_pred_proba = y_train_pred_proba[:, 1]

    # predict test probabilities
    y_test_pred_proba = model.predict_proba(x_test)
    y_test_pred_proba = y_test_pred_proba[:, 1]

    # plot roc curve
    plot_roc_curve(y_test, y_test_pred_proba)

    #plot_find_max_fscore
    opt_cutoff = find_max_fscore(y_test, y_test_pred_proba)

    #plot_confusion_matrix
    plot_confusion_matrix(y_test, y_test_pred_proba, opt_cutoff)

    #plot_feature_importances
    plot_feature_importnaces(model,x_train)

    #calculate metrics
    gini_train = 2 * roc_auc_score(y_train, y_train_pred_proba) - 1
    gini_test = 2 * roc_auc_score(y_test, y_test_pred_proba) - 1
    f_score_train = f1_score(y_train, (y_train_pred_proba > opt_cutoff), pos_label=1, average='binary')
    f_score_test  = f1_score(y_test , (y_test_pred_proba  > opt_cutoff), pos_label=1, average='binary')

    #print metrics
    print('gini_train:', gini_train)
    print('gini_test:' , gini_test)
    print('f_score_train:', f_score_train )
    print('f_score_test', f_score_test)

    model.save_model(model_name)

def model_inference(df_proc, df_event, cutoff, model_name):

    #load model
    model = CatBoostClassifier()
    model.load_model(model_name)

    #make_prediction probabilities
    pred_proba = model.predict_proba(df_proc)
    pred_proba = pred_proba[:, 1]

    #make prediction
    prediction = (pred_proba > cutoff).astype(int)
    df_event['target'] = prediction

    return df_event

def save_result(df,filepath_output):

    df.to_excel(os.path.join(filepath_output,'result.xlsx'))