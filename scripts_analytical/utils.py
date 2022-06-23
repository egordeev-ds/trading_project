import os
import itertools

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_curve, confusion_matrix
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
      
def plot_candletick_anomaly(df):
    
    fig = make_subplots()
    
    fig.add_scatter(
        x = df[df['true'] == True]['t_start'],
        y = df[df['true'] == True]['open_price'],
        mode = 'markers',
        marker=dict(size=5, color="blue"),
        name = 'true'
    )
    
    fig.add_scatter(
        x = df[df['predicted'] == True]['t_start'],
        y = df[df['predicted'] == True]['close_price'],
        mode = 'markers',
        marker=dict(size=5, color="purple"),
        
        name='predicted'
    )

    fig.add_trace(
        go.Candlestick(
            x=df['t_start'],
            open=df['open_price'],
            high=df['high_price'],
            low=df['low_price'],
            close=df['close_price'],
            name='klines'
        )
    )
    
    fig.show()

def plot_roc_curve(y_true, y_score):
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    plt.plot([0,1], [0,1], linestyle='--', label = 'No Skill')
    plt.plot(fpr, tpr, marker='.', label = 'CatBoost')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend()

    plt.show()

def find_max_fscore(y_true, y_score):

    f_score = []
    cutoff_list = np.arange(0,1,0.01)
 
    for cutoff in cutoff_list:
        y_pred = (y_score > cutoff).astype(int)
        f_scr = f1_score(y_true, y_pred, pos_label=1, average='binary')
        f_score.append(f_scr)
       
    f_score = pd.Series(f_score, index=cutoff_list)

    return f_score.idxmax()

def find_max_recall(y_true, y_score):

    rec_score = []
    cutoff_list = np.arange(0,1,0.01)
 
    for cutoff in cutoff_list:
        y_pred = (y_score > cutoff).astype(int)
        rec_scr = recall_score(y_true, y_pred, pos_label=1, average='binary')
        rec_score.append(rec_scr)
       
    rec_score = pd.Series(rec_score, index=cutoff_list)

    return rec_score.idxmax()

def plot_confusion_matrix(y_true, y_score, cutoff):
    
    cm = confusion_matrix(y_true, y_score > cutoff)
    classes = ['Non-event', 'event']
    
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

def plot_feature_importnaces(model,x_train,top = 20):
    
    importances = model.get_feature_importance()
    forest_importances = pd.Series(importances, index=x_train.columns.to_list())
    forest_importances = forest_importances.sort_values(ascending = False)[:top]
    
    fig, ax = plt.subplots()
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances")
    fig.tight_layout()
    plt.show()