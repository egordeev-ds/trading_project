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
    vol_per_trade = []

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
                vol_per_trade.append(float(
                    js_i['r']['data']['k']['q'])/js_i['r']['data']['k']['n'])
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
        'vol_per_trade':vol_per_trade,
        'is_closed':x,
        }

    df = pd.DataFrame(data=d)

    return df

def generate_features(df, feature_cols, shift_backwards, shift_forward, anomaly_crtiretion):
    
    ##deltas
    for col in feature_cols:
        df[f'{col}_delta'] = df[col].shift(-1) - df[col]    

    #correction
    condition = (df['is_closed'] == True)
    feature_cols_delta = [col + '_delta' for col in feature_cols]
    for col_1, col_2 in zip(feature_cols,feature_cols_delta):
        df[col_2].loc[condition] = (df[col_1].loc[condition].values) + (df[col_2].loc[condition].values)

    ##shift deltas
    data_temp = pd.DataFrame()
    for col in feature_cols_delta:
        for shift in range(shift_backwards+1):
            #deltas
            data_temp[f'{col}_{shift}'] = df[col].shift(shift)
            #cumsum deltas
            if shift > 1:
                data_temp[f'{col}_0_{shift-1}'] = df[col].rolling(shift).sum()

    df = pd.concat([df.drop(feature_cols_delta, axis = 1), data_temp], axis = 1)
    df = df.dropna().reset_index(drop = True)

    #target calculation
    df['anomaly_t_start'] = np.where(
        df.close_price > df.open_price,
        df.high_price/df.open_price,
        df.open_price/df.low_price
        )
    df['anomaly_t_end'] = df['anomaly_t_start'].shift(-shift_forward)
    df['target'] = df['anomaly_t_end'] > anomaly_crtiretion
    df = df.dropna()

    return df

def process_features(df, feature_cols):

    #correction_1
    df['t_start'] = df['event_time']
    df['t_end'] = df['event_time'].shift(-1)

    df = df.dropna()
    df['event_time'] = pd.to_datetime(df['event_time'].apply(
        lambda x: datetime.fromtimestamp(x/1000.0)
        ))
    df['t_start'] = pd.to_datetime(df['t_start'].apply(
        lambda x: datetime.fromtimestamp(x/1000.0)
        ))
    df['t_end'] = pd.to_datetime( df['t_end'].apply(
        lambda x: datetime.fromtimestamp(x/1000.0)
        ))

    #correction_2
    df = df.drop_duplicates('event_time', keep = 'first')
    df = df.sort_values(by='event_time').reset_index(drop = True)
    df = df.drop('event_time' , axis = 1)
    df = df.round(2)
    df = df.reset_index(drop = True)

    #select
    dt_cols = ['t_start', 't_end']
    feature_cols = feature_cols
    delta_cols = [col for col in df.columns if 'delta' in col]

    #select_состояние
    df_event = df[dt_cols + feature_cols + ['target']]

    #select_изменение_состояния
    df_period = df[dt_cols + delta_cols + ['target']]

    #select_обучаяющая_выборка
    df_model = df[delta_cols + ['target']]
    
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
    prediction = pred_proba > cutoff
    df_event['target'] = prediction

    return df_event

def save_result(df,filepath_output):

    df.to_excel(os.path.join(filepath_output,'result.xlsx'))