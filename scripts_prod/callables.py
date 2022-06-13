#imports
import os
import json
import itertools
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score,roc_auc_score,roc_curve,confusion_matrix
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

def prepare_dataset(filepath_input):

    #parsing json

    files=os.listdir(filepath_input)
    kline_js=[]
    for name in tqdm(files):
        f=open(filepath_input+name)
        js=json.load(f)
        for i in tqdm(range(len(js))):
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

    df['t_start'] = df['event_time']
    df['t_end'] = df['event_time'].shift(-1)

    df['close_delta'] = df['close_price'].shift(-1) - df['close_price']
    df['trades_delta'] = df['n_trades'].shift(-1) - df['n_trades']

    df['quote_volume_delta'] = df['quote_volume'].shift(-1) - df['quote_volume']
    df['base_volume_delta'] = df['base_volume'].shift(-1) - df['base_volume']

    df['buy_quote_delta'] = df['buy_quote'].shift(-1) - df['buy_quote']
    df['buy_base_delta'] = df['buy_base'].shift(-1) - df['buy_base']

    df = df.drop_duplicates('event_time', keep = 'first')
    df = df.sort_values(by='event_time').reset_index(drop = True)
    df = df.dropna()

    #save to hdfs
    #df.to_csv(filepath_output)

    return df

def process_dataset(df, shift, anomaly_crtiretion):

    #drop event_time
    df = df.drop('event_time' , axis = 1)

    #change type
    df.t_start = pd.to_datetime(df.t_start)
    df.t_end = pd.to_datetime(df.t_end)

    #correction
    df['trades_delta'].loc[df['is_closed'] == True] = (df[df['is_closed'] == True]['n_trades'].values) + (df[df['is_closed'] == True]['trades_delta'].values)
    df['quote_volume_delta'].loc[df['is_closed'] == True] = (df[df['is_closed'] == True]['quote_volume'].values) + (df[df['is_closed'] == True]['quote_volume_delta'].values)
    df['base_volume_delta'].loc[df['is_closed'] == True] = (df[df['is_closed'] == True]['base_volume'].values) + (df[df['is_closed'] == True]['base_volume_delta'].values)
    df['buy_quote_delta'].loc[df['is_closed'] == True] = (df[df['is_closed'] == True]['buy_quote'].values) + (df[df['is_closed'] == True]['buy_quote_delta'].values)
    df['buy_base_delta'].loc[df['is_closed'] == True] = (df[df['is_closed'] == True]['buy_base'].values) + (df[df['is_closed'] == True]['buy_base_delta'].values)

    #аномалия на начало периода
    df['anomaly_t_start'] = np.where(df.close_price > df.open_price,
                                    df.high_price/df.open_price,
                                    df.open_price/df.low_price)

    #аномалия на конец периода
    df['anomaly_t_end'] = df['anomaly_t_start'].shift(shift)

    #target - на конец периода (спустя 250мс после t_start)
    df['target'] = df['anomaly_t_end'] > anomaly_crtiretion

    #select_состояние
    df_event = df[['t_start','open_price','close_price','low_price','high_price','n_trades','base_volume','quote_volume','buy_base','target']]

    #select deltas
    df_proc = df[['t_start','t_end','close_delta','trades_delta','quote_volume_delta','buy_quote_delta','target']]

    return df_event, df_proc
    
def plot_candletick(df):
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Candlestick(
            x=df['event_time'],
            open=df['open_price'],
            high=df['high_price'],
            low=df['low_price'],
            close=df['close_price']
        ))
    
    fig.show()
      
def plot_candletick_anomaly(df,filepath_output, left_b, right_b):

    df = df[left_b:right_b]
        
    fig = px.scatter(
        x = df[df['target'] == 1]['t_start'],
        y = df[df['target'] == 1]['open_price']
    )

    fig.add_trace(
        go.Candlestick(
            x=df['t_start'],
            open=df['open_price'],
            high=df['high_price'],
            low=df['low_price'],
            close=df['close_price']
        ))
    
    fig.show()

    fig.write_image(os.path.join(filepath_output,'candlestick.png'))

# plot roc curve
def plot_roc_curve(y_true, y_score):
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    plt.plot([0,1], [0,1], linestyle='--', label = 'No Skill')
    plt.plot(fpr, tpr, marker='.', label = 'CatBoost')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend()

    plt.show()

#find_max_fscore
def find_max_fscore(y_true, y_score):

    f_score = []
    cutoff_list = np.arange(0,1,0.01)
 
    for cutoff in cutoff_list:
        y_pred = (y_score > cutoff).astype(int)
        f_scr = f1_score(y_true, y_pred, pos_label=1, average='binary')
        f_score.append(f_scr)
       
    f_score = pd.Series(f_score, index=cutoff_list)

    fig, ax = plt.subplots()

    f_score.plot(ax=ax)

    plt.show()
 
    return f_score.idxmax()

#plot_confusion_matrix
def plot_confusion_matrix(y_true, y_score, cutoff):
    
    cm = confusion_matrix(y_true, y_score > cutoff)
    classes = ['Non-anomaly', 'anomaly']
    
    fig, ax = plt.subplots()
    
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    ax.set_title('Confusion matrix')
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    
    ax.set_xticks(np.arange(len(classes)), classes, rotation=45)
    ax.set_yticks(np.arange(len(classes)), classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    fig.set_size_inches(8, 5)
    fig.set_tight_layout(True)
    
    plt.show()

#plot_feature_importnaces
def plot_feature_importnaces(model,x_train):
    
    importances = model.get_feature_importance()
    
    forest_importances = pd.Series(importances, index=x_train.columns.to_list())
    forest_importances = forest_importances.sort_values(ascending = False)

    fig, ax = plt.subplots()

    forest_importances.plot.bar(ax=ax)

    ax.set_title("Feature importances")

    fig.tight_layout()

    plt.show()

def model_build(data, model_name):

    data = data.drop(['t_start', 't_end'], axis = 1)

    #split
    train, test = train_test_split(data, test_size = 0.2, random_state = 42, shuffle = False)

    x_train = train.drop('target', axis = 1)
    y_train = train.target.astype(int)

    x_test = test.drop('target', axis = 1)
    y_test = test.target.astype(int)

    #fit best model
    params = {
        'iterations': 300,
        'learning_rate': 0.03,
        'depth': 5,
        'l2_leaf_reg': 2,
        'rsm': 0.7,
        'verbose': False,
        'allow_writing_files': False,
        'random_state': 42
    }
    model = CatBoostClassifier(**params)
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
    f_score_train = f1_score(y_train, (y_train_pred_proba > opt_cutoff))
    f_score_test  = f1_score(y_test , (y_test_pred_proba  > opt_cutoff))

    #print metrics
    print('gini_train:', gini_train)
    print('gini_test:' , gini_test)
    print('f_score_train:', f_score_train )
    print('f_score_test', f_score_test)

    model.save_model(model_name)

def model_inference(df_proc,df_event,model_name):

    #load model
    model = CatBoostClassifier()
    model.load_model(model_name)

    #make_prediction probabilities
    pred_proba = model.predict_proba(df_proc)
    pred_proba = pred_proba[:, 1]

    #make prediction
    cutoff = 0.04
    prediction = (pred_proba > cutoff).astype(int)
    df_event['target'] = prediction

    return df_event

def save_result(df,filepath_output):

    df.to_excel(os.path.join(filepath_output,'result.xlsx'))