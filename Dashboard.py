import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import dash_table
from dash.dependencies import Input, Output, State

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from xgboost import XGBRegressor
from sklearn.decomposition import PCA
import pickle

from cleaning_data import clean

loadModel = pickle.load(open('Housing_prices_xgb.sav', 'rb'))


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

train = pd.read_csv('train.csv').drop(['Id'],axis=1)
y = train['SalePrice']
train.drop(['SalePrice'],axis=1,inplace=True)
test = pd.read_csv('test.csv').drop(['Id'],axis=1)
cleaned_data = pd.read_csv('Cleaned_data.csv').drop(['Id'],axis=1)
final_cleaned_data = pd.read_csv('Final_cleaned_data.csv').drop(['Id'],axis=1)

coef1 = pd.Series(loadModel.feature_importances_, final_cleaned_data.columns).sort_values(ascending=False)
coef1 = pd.DataFrame(coef1,columns=['Feature Importance'])

def classclassifier(colname):
    if train[colname].dtypes == 'object':
        uniques = train[colname].unique()
        uniques = [uniques for uniques in uniques if str(uniques) != 'nan']
        return html.Div(children = [
            dcc.Dropdown(id='{}'.format(colname),
            options = [{'label':i, 'value':i} for i in uniques],
            value=train[colname].mode()[0]
            )
        ],className='col-3')
    elif train[colname].dtypes in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
        return html.Div(children = [
            dcc.Input(id='{}'.format(colname),type='number',value=train[colname].iloc[0])
        ],className='col-3')

def loop_input(colname):
    return html.Div(children=[
        html.Div(children = [
            html.H5('{}: '.format(colname))
        ],className='col-3'),
        classclassifier(colname)
    ], className='row')

divs = []
for colname in train.columns:
    divs.append(loop_input(colname))

# def loop_input(colname):
#     return html.Div([html.H5('X{}'.format(x))])

# divs = []
# for colname in train.select_dtypes('object'):
#     divs.append(loop_input(colname))

app.layout = html.Div(children = [
        html.H1('House Prices'),
        dcc.Tabs(value='tabs',id='tabs-1',children = [
            dcc.Tab(label='Table',value='tabfour',children=[
                html.Div(id='pokemontable',children =
                    dash_table.DataTable(
                        id='table',
                        columns=[{"name": i, "id": i} for i in cleaned_data.columns],
                        data=cleaned_data.to_dict('records'),
                        page_action = 'native',
                        page_current = 0,
                        page_size = 10,
                        style_table={'overflowX': 'scroll'}
                    )
                )
            ]),

            dcc.Tab(label='Feature Importance',value='Feature Importance',children = [
                html.Div([
                    html.Div(children=[html.H5('Number of Top Features')],className='col-3'),
                    html.Div(children=[
                        dcc.Dropdown(id='x1',
                        options = [{'label':i, 'value':i} for i in range(len(final_cleaned_data.columns))],
                        value = 10
                        )
                    ],className='col-3')
                    
                ],className='row'),
                html.Div([
                    dcc.Graph(
                    id = 'contoh-graph-bar',
                    figure={
                    'data':[
                        {'x': list(coef1.index),'y':coef1['Feature Importance'],'type':'bar','name':'Feature Importance'}
                    ],
                        'layout':{'title':'Feature Importance'}
                    }
                        )])
            ]),

            dcc.Tab(label='Price Prediction',value='tabone',children = [
                html.Div(className='col-6',children=divs),
                html.Div(children=[
                    html.Button('Predict', id='predict_price')
                ]),
                html.Div(id='prediction_output',children=[])
            ])
        ])
])


def retrieve_input(colname):
    return State(component_id = '{}'.format(colname),component_property = 'value')

states_values = []
for colname in train.columns:
    states_values.append(retrieve_input(colname))

@app.callback(
    Output(component_id='prediction_output', component_property='children'),
    [Input(component_id = 'predict_price', component_property='n_clicks')],
    states_values
)
def results(*args):
    input_names = [states.component_id for states in states_values]
    kwargs_dict = dict(zip(input_names, args))
    new_data = pd.DataFrame(kwargs_dict,index=['new'])
    new_data = new_data[~new_data.isnull()]
    new_data_cleaned = clean(new_data,pd.concat([train,test]))
    xgb = XGBRegressor(learning_rate=0.01,
                    n_estimators= 3460,
                    max_depth= 3, 
                    min_child_weight= 0,
                    subsample= 0.7,
                    colsample_bytree= 0.7,
                    nthread= -1)
    xgb.fit(new_data_cleaned.iloc[:len(train),:],y)

    predict_target = xgb.predict(new_data_cleaned.iloc[len(train):,:])
    return  html.H3("Precited House Price: ${}".format(predict_target[0]))


@app.callback(
    Output(component_id='contoh-graph-bar', component_property='figure'),
    [Input(component_id = 'x1', component_property='value')]
)
def create_graph(x1):
    figure={
        'data':[
            {'x': list(coef1.iloc[:x1].index),'y':coef1.iloc[:x1]['Feature Importance'],'type':'bar','name':'Feature Importance'}
        ],
            'layout':{'title':'Feature Importance'}
        }
    return figure 

if __name__ == '__main__':
    app.run_server(debug=True)
