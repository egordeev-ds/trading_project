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

def parse_json(filepath_input):

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
                l.append(float(js_i['r']['data']['k']['l']))
                h.append(float(js_i['r']['data']['k']['h']))
                v.append(float(js_i['r']['data']['k']['v']))
                q.append(float(js_i['r']['data']['k']['q']))
                V.append(float(js_i['r']['data']['k']['V']))
                Q.append(float(js_i['r']['data']['k']['Q']))
                n.append(js_i['r']['data']['k']['n'])
                x.append(js_i['r']['data']['k']['x'])
                
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
        'is_closed':x,
        }

    df = pd.DataFrame(data=d)

    return df

def process_data(df):

    #correction_1
    df['event_time'] = pd.to_datetime(
        df['event_time'].apply(
            lambda x: datetime.fromtimestamp(x/1000.0)
        )
    )
    df['t_start'] = df['event_time']
    df['t_end'] = df['event_time'].shift(-1)
    df = df.dropna()

    #correction_2
    df = df.drop_duplicates('event_time', keep = 'first')
    df = df.sort_values(by='event_time').reset_index(drop = True)
    df = df.drop('event_time' , axis = 1)
    df = df.dropna()

    #correction_3
    df['buy_base'] = df['buy_base']*1000000
    df = df.drop(['quote_volume','buy_quote'], axis = 1)
    
    return df

def generate_features(df, shift_b, shift_f, anomaly_crtiretion):
    
    feature_cols = ['close_price','base_volume','buy_base','n_trades']

    #deltas
    for col in feature_cols:
        df[f'{col}_delta'] = df[col].shift(-1) - df[col]
    condition = (df['is_closed'] == True)
    delta_cols = [col + '_delta' for col in feature_cols]
    for col_1, col_2 in zip(feature_cols,delta_cols):
        df[col_2].loc[condition] = (df[col_1].loc[condition].values) + (df[col_2].loc[condition].values)
    df = df.drop(axis=0, index = [1774,79063])
    df = df.drop('is_closed', axis = 1)

    #vol_per_trade
    df['vol_per_trade_delta'] = df['base_volume_delta'] / df['n_trades_delta']
    delta_cols.append('vol_per_trade_delta')

    #shift deltas
    data_temp = pd.DataFrame()
    for col in delta_cols:
        for shift in range(1,shift_b+1):
            data_temp[f'{col}_{shift}'] = df[col].shift(shift)
    df = pd.concat([df, data_temp], axis = 1)

    #shift cumsum deltas
    data_temp = pd.DataFrame()
    for col in delta_cols:
        for shift in range(2,shift_b+1):
            data_temp[f'{col}_0_{shift-1}'] = df[col].rolling(shift).sum()
    df = pd.concat([df, data_temp], axis = 1)

    #target
    df['anomaly_t_start'] = np.where(
        df.close_price > df.open_price,
        df.high_price/df.open_price,
        df.open_price/df.low_price
        )
    df['anomaly_t_end'] = df['anomaly_t_start'].shift(-shift_f)
    df['target'] = df['anomaly_t_end'] > anomaly_crtiretion

    return df

def select_features(df):

    #columns
    dt_cols = ['t_start', 't_end']
    price_cols = ['open_price','close_price','low_price','high_price']
    feature_cols = ['base_volume','buy_base','n_trades']
    target_cols = ['anomaly_t_start','anomaly_t_end','target']
    delta_cols = df.drop(dt_cols+price_cols+feature_cols+target_cols, axis = 1).columns.to_list()

    ###prepare
    df = df.dropna()
    df = df.reset_index(drop = True)

    ###select
    df_event = df[dt_cols + price_cols + ['base_volume'] + ['target']]
    df_model = df[delta_cols + ['target']]

    return df_event, df_model

def model_train(data, cb_params):

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

def model_inference(df_proc, df_event, cutoff, model_name):

    #load model
    model = CatBoostClassifier()
    model.load_model(model_name)

    #make_prediction probabilities
    pred_proba = model.predict_proba(df_proc)
    pred_proba = pred_proba[:, 1]

    #make prediction
    prediction = pred_proba > cutoff
    df_event['target'] = prediction

    return df_event

def save_result(df,filepath_output):

    df.to_excel(os.path.join(filepath_output,'result.xlsx'))