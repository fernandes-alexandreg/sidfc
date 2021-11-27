import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from statistics import NormalDist

data = pd.read_pickle("Data/forecast_1_2017.pkl").sort_index()

app = dash.Dash(__name__)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(html.Div(
    
    [
        
    html.Div([
           
        dbc.Row([
            
                html.Div(html.H1("Demand Forecasting: Visualizing Results"))
         ])
    ]),
        
    html.Div([
        
        
         dbc.Row(
            [
                dbc.Col(html.Div(dcc.RadioItems
                (
                    id='agg_func_radio',
                    options=[{'label': i, 'value': i[:-1]} for i in ['max ', 'mean ','min ']],
                    value='max',
                    labelStyle={'display': 'inline-block'}
                )), width=4),
                
                dbc.Col(html.Div(dcc.DatePickerRange(
                    id='date-picker-range',
                    min_date_allowed = data.index.min(),
                    max_date_allowed = data.index.max(),
                    initial_visible_month= data.index.min(),
                    start_date = data.index.min(),
                    end_date= data.index.max()
                )), width=4),
            ],
            justify="between",
        ),
        
    ]),
        
    
    html.Div([
        
        dbc.Row(
            [
            
            dbc.Col(html.Div(dcc.Graph(
                id='forecast-descriptives',
                  hoverData = {'points': [{'curveNumber': 0, 'x': 1, 'y': 1}]}
            )), width=3),
        
           dbc.Col(html.Div(dcc.Graph(
               id='forecast-history'
           )), width={"size": 9, "offset": 0}),
                
        ],
        justify="start",
            
        ),
    ]),
        
    html.Div([
        
        dbc.Row(
            [
                
                dbc.Col(html.Div(dcc.Slider(
                    id='confidence_slider',
                    min= 0,
                    max= 99.9999,
                    value= 99.9999,
                    step=1,
                    tooltip={"placement": "top", "always_visible": True}
            )), width=6),
        ],
        justify="end",
            
        ), 
    ])
    
]))


@app.callback(
    Output('forecast-descriptives', 'figure'),
    Output('forecast-history', 'figure'),
    Input('agg_func_radio', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
    Input('forecast-descriptives', 'hoverData'),
    Input('confidence_slider', 'value')
)

def update_graph(agg_func,
                 start_date,
                 end_date,
                 descriptives_hoverData,
                 confidence_slider_value):
    
    store = descriptives_hoverData['points'][0]['x']
    item = descriptives_hoverData['points'][0]['y']
    
    p = pd.pivot_table(data[start_date:end_date],values="mean_forecast", columns = 'store', index = 'item', aggfunc= agg_func)
    
    fig1 = px.imshow(p,
                labels=dict(x="Store", y="Item", color="Sales"),
                zmax = 180,
                color_continuous_scale=px.colors.sequential.Jet
                    )
    
    fig1.layout.coloraxis.showscale = False
    
    fig1.update_layout(
                        margin=dict(l=100, r=100, t=30, b=30),
                        paper_bgcolor="LightSteelBlue",
                      )

    selection = data.loc[(data.store == store) &
                         (data.item == item)]

    confidence = confidence_slider_value / 100
    
    z = NormalDist().inv_cdf((1 + confidence) / 2.)

    fig2 = go.Figure([
        go.Scatter(
            name='Mean',
            x=selection.index,
            y=selection.mean_forecast,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='Upper Bound',
            x= selection.index,
            y= selection['mean_forecast'] + z*selection['std'],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x= selection.index,
            y= selection['mean_forecast'] - z*selection['std'],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    fig2.update_layout(
        yaxis_title='Sales',
        hovermode="x",
        xaxis_range=[start_date,end_date],
        yaxis_range=[0,180]
    )
    
    fig2.update_layout(
                        margin=dict(l=20, r=20, t=20, b=20),
                        paper_bgcolor="LightSteelBlue",
                      )
    
    
    return fig1, fig2

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port='80')