#########################
# Tab 1: Data Exploration
##########################

from dash import Dash
from dash import html, dcc, Output, Input
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
url = 'https://raw.githubusercontent.com/ananyaprakash/ChatGPT_Twitter_Analysis/main/ChatGPT_tweets_processed_50k.csv'
df= pd.read_csv(url)
df.reset_index(inplace=True)
# df_red = df.iloc[50000:]
# df_red.to_csv('ChatGPT_tweets_processed_50k.csv', index=False)
print(df.head(10))
features= ['like_count' , 'retweet_count',  'content_length','hashtag_count','tagged_users_count']
#%%
filtered_df = df.iloc[:2000]
print(filtered_df.shape)
    #%%
print(df.shape)
df_date_grouped = df.groupby(['date'])['date'].count()
print(df_date_grouped.head(10))

#%%
# fig = px.line(df, x= df.index, y='content_length')
# fig.show(renderer ='browser')
#%%
my_app= Dash('My app', external_stylesheets= external_stylesheets)

# Define app layout
my_app.layout = html.Div([

    html.H1("Raw data exploration"),
    html.Div([
        html.Label("Select feature names"),
        dcc.Dropdown(
            id='feature-dropdown',
            options=[{'label': feature, 'value': feature} for feature in features],
            multi=True,
            value=['content_length'],  # Default selected values
        ),
        html.Br(),
        html.Label("Please select graph mode"),
        dcc.RadioItems(
            id='graph-mode',
            options=['lines', 'markers'],
            value='lines'
        ), html.Br(),
    ], style={'margin-bottom': '20px'}),

    dcc.Graph(id='graph-raw'),
    html.Br(),
    html.H4('Pick number of observations'),
    dcc.Slider(
        id='num_observations',
        min= 5000,
        max=53452,
        step=5000,
        value=15000,
        marks={i: str(i) for i in range(5000,df.shape[0]+1,5000 )},
    ),
    html.Div([
        html.Label("Select feature names"),
        dcc.Graph(id='graph-hist'),
        html.Br(),
    ], style={'margin-bottom': '20px'}),
    # dcc.Graph(id='graph'),
])

# Define callback to update the plot based on dropdown selection
@my_app.callback(
    Output('graph-raw', 'figure'),
        Output('graph-hist', 'figure'),
    [Input('feature-dropdown', 'value'),
     Input('num_observations', 'value'),
     Input('graph-mode', 'value'),
     ]
)
def update_plot(selected_features, num_observations, mode):
    traces1 = []
    traces2 = []
    # Plot f(x)
    # fig1 = go.Figure()
    filtered_df = df.iloc[:num_observations]
    print(filtered_df.head())
    for feature in selected_features:
        trace1 = go.Scatter(x=filtered_df.index, y=filtered_df[feature], mode=mode, name=feature)
        trace2 = go.Histogram(x=filtered_df[feature], autobinx=True, opacity=0.7 )
        traces1.append(trace1)
        traces2.append(trace2)

    # Create a layout for the plot
    layout = go.Layout(
        title='Raw Data Visualization',
        xaxis=dict(title='Index'),
        yaxis=dict(title='Magnitude'),
        template='plotly_dark'
    )

    # Create a figure and add the traces to it
    fig1 = go.Figure(data=traces1, layout=layout)

    # fig1.add_trace(go.Scatter( x= filtered_df.index, y=filtered_df[selected_features], mode='markers'))
    # fig1.update_layout(title='Raw Data of selected features', xaxis_title='index', yaxis_title='magnitude')

    # Plot FFT
    # f_values_fft = fft(f_values)
    # freq = np.fft.fftfreq(samples, d=x_values[1] - x_values[0])
    # fig2 = go.Figure()
    fig2 = go.Figure(data=traces2)
    fig2.update_layout(title='Histogram plots of above generated data', xaxis_title='Values',
                       yaxis_title='Frequency', template='plotly_dark')
    # for feature in selected_features:
    #     trace = dict(
    #         x=df.index[:num_observations],
    #         y=df[feature][:num_observations],
    #         mode='lines',
    #         name=feature
    #     )
    #     traces.append(trace)

    # layout = dict(title=f'Line Plot of numerical feature values of {" ".join(selected_features)}', xaxis=dict(title='post id'), yaxis=dict(title='Values'))

    # return dict(data=traces, layout=layout)
    return fig1,fig2
# Run the app
if __name__ == '__main__':
    my_app.run_server(debug=True)