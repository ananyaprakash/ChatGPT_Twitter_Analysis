import math



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


my_app = Dash('my app', external_stylesheets=external_stylesheets)
my_app.layout = html.Div(
[html.H3("Homework 4", style={"textAlign":"center"}),
dcc.Tabs(id='hw-questions',
children=[
dcc.Tab(label='Tab1 ', value='q1'),
dcc.Tab(label='Tab2 ', value='q2'),
# dcc.Tab(label='Question3 ', value='q3'),
# dcc.Tab(label='Question4 ', value='q4'),
# dcc.Tab(label='Question5 ', value='q5'),
# dcc.Tab(label='Question6 ', value='q6')
]),
html.Div(id = 'layout')
#
]
)
# # ================
# # Question 1
# # ===============
question1_layout = html.Div([

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

    filtered_df = df.iloc[:num_observations]
    print(filtered_df.head())
    for feature in selected_features:
        trace1 = go.Scatter(x=filtered_df.index, y=filtered_df[feature], mode=mode, name=feature)
        trace2 = go.Histogram(x=filtered_df[feature], autobinx=True, opacity=0.7 )
        traces1.append(trace1)
        traces2.append(trace2)

    layout = go.Layout(
        title='Raw Data Visualization',
        xaxis=dict(title='Index'),
        yaxis=dict(title='Magnitude'),
        template='plotly_dark'
    )

    fig1 = go.Figure(data=traces1, layout=layout)
    fig2 = go.Figure(data=traces2)
    fig2.update_layout(title='Histogram plots of above generated data', xaxis_title='Values',
                       yaxis_title='Frequency', template='plotly_dark')

    return fig1,fig2

# # ================
# # Question 2
# # ===============
question2_layout= html.Div(children=[
    html.H1(children='Outlier Detection'),

    # Checklist for selecting features
    dcc.Checklist(
        id='feature-selector',
        options=[
            {'label': 'like_count', 'value': 'feature1'},
            {'label':  'retweet_count', 'value': 'retweet_count'},
            {'label': 'content_length', 'value': 'content_length'},
            {'label': 'hashtag_count', 'value': 'hashtag_count'},
            {'label': 'tagged_users_count', 'value': 'tagged_users_count'},

        ],
        value=['hashtag_count'],  # Default selection
        inline=True,

    ),
    html.Br(),

    # Dropdown menu for selecting plot type
    dcc.Dropdown(
        id='plot-type-selector',
        options=[
            {'label': 'Box Plot', 'value': 'box'},
            {'label': 'Violin Plot', 'value': 'violin'},
        ],
        value='box',  # Default selection
        style={'width': '50%'}
    ),
    html.Br(),
    # Display the selected plot
    dcc.Graph(id='plot-display', style={'width': '50%'})
])


# Callback to update the displayed plot based on user selections
@my_app.callback(
    Output('plot-display', 'figure'),
    [Input('feature-selector', 'value'),
     Input('plot-type-selector', 'value')]
)
def update_plot(selected_features, selected_plot_type):
    if not selected_features or not selected_plot_type:
        return px.scatter()  # Return an empty plot if no features or plot type selected


    # Create the selected plot type (box, violin, or swarm)
    if selected_plot_type == 'box':
        fig = px.box(df, y=selected_features)
        fig.update_layout(
            template='plotly_dark',
            height=800  # Set your desired height
        )
        return fig
    elif selected_plot_type == 'violin':
        fig = px.violin(df, y=selected_features)
        fig.update_layout(
            template='plotly_dark',
            height=800  # Set your desired height
        )
        return fig










# url="https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/CONVENIENT_global_confirmed_cases.csv"
# df = pd.read_csv(url)
# df.rename(columns={'Country/Region': 'Date'}, inplace=True)
# df.drop(index=0, axis=0, inplace=True)
# # print(df.head())
# ########## aggregate sum columns ########
# # print(df.columns.to_list())
# sum_cols=[ x for x in df.columns.to_list() if x.startswith('China')]
# df['China_sum']=df[sum_cols].astype(float).sum(axis=1)
# sum_cols=[ x for x in df.columns.to_list() if x.startswith('United Kingdom')]
# df['United_Kingdom_sum']=df[sum_cols].astype(float).sum(axis=1)
# countries=['US', 'Brazil', 'United_Kingdom_sum', 'China_sum','India', 'Italy','Germany']
#
# question1_layout = html.Div([
#     html.Div([ html.H1("COVID Global confirmed Cases"), ],
#              style={'text-align': 'center'}),
#
#
#     html.Label("Select country names"),
#     dcc.Dropdown(
#         id='country-dropdown',
#         options=[{'label': country, 'value': country} for country in countries],
#         multi=True,
#         value=['US'],  # Default selected values
#     ),
#
#     dcc.Graph(id='graph'),
# ]
# )
# @my_app.callback(
#     Output('graph', 'figure'),
#     [Input('country-dropdown', 'value')]
# )
#
# def update_q1(selected_countries):
#     traces = []
#     for country in selected_countries:
#         trace = dict(
#             x=df['Date'],
#             y=df[country],
#             mode='lines',
#             name=country
#         )
#         traces.append(trace)
#
#     layout = dict(title=f'Line Plot of confirmed COVID cases for {" ".join(selected_countries)}', xaxis=dict(title='Date'), yaxis=dict(title='Values'))
#
#     return dict(data=traces, layout=layout)
#
#
# # ================
# # Question 2
# # ===============
# question2_layout = html.Div([
#     html.Div([ html.H1("Quadratic Function Plotter"),],
#              style={'text-align': 'center'}),
#
#
#     html.Div([
#         html.Label("Coefficient a"),
#         dcc.Slider(
#             id='coeff-a',
#             min=-10,
#             max=10,
#             step=0.5,
#             value=1,
#             marks={i: f'{i:.1f}' for i in np.arange(-10, 11, 0.5)},
#         ),
#     ], style={'margin-bottom': '20px'}),
#
#     html.Div([
#         html.Label("Coefficient b"),
#         dcc.Slider(
#             id='coeff-b',
#             min=-10,
#             max=10,
#             step=0.5,
#             value=0,
#             marks={i: f'{i:.1f}' for i in np.arange(-10, 11, 0.5)},
#         ),
#     ], style={'margin-bottom': '20px'}),
#
#     html.Div([
#         html.Label("Coefficient c"),
#         dcc.Slider(
#             id='coeff-c',
#             min=-10,
#             max=10,
#             step=0.5,
#             value=0,
#             marks={i: f'{i:.1f}' for i in np.arange(-10, 11, 0.5)},
#         ),
#     ], style={'margin-bottom': '20px'}),
#
#     dcc.Graph(id='quadratic-plot'),
# ])
#
# @my_app.callback(
#     Output('quadratic-plot', 'figure'),
#     [Input('coeff-a', 'value'),
#      Input('coeff-b', 'value'),
#      Input('coeff-c', 'value')]
# )
# def update_plot(a, b, c):
#     x_values = np.linspace(-2, 2, 1000)
#     y_values = (a * x_values**2) + (b * x_values) + c
#
#     fig = px.line(x=x_values, y=y_values, labels={'x': 'x', 'y': 'f(x)'}, title='Quadratic Function Plot $f(x) = axˆ2+bx+c$')
#     return fig
#
# # ================
# # Question 3
# # ===============
# question3_layout = html.Div([
#    html.Div([html.H1("Calculator")],
#             style={'text-align':'center'}) ,
#
#     html.Div([
#         html.Label("Please enter the first number"),
#         html.Br(),
#         html.Label("input"),
#         dcc.Input(id='input-a', type='number', value=0),
#     ], style={'margin-bottom': '20px'}),
#
#
#     html.Div([
#         html.Label("Select Operation:"),
#         dcc.Dropdown(
#             id='operation-dropdown',
#             options=[
#                 {'label': '+', 'value': 'add'},
#                 {'label': '-', 'value': 'subtract'},
#                 {'label': 'Multiplication', 'value': 'multiply'},
#                 {'label': 'Division', 'value': 'divide'},
#                 {'label': 'Logarithm', 'value': 'log'},
#                 {'label': 'Root', 'value': 'root'},
#             ],
#             value='add',
#         ),
#     ], style={'margin-bottom': '20px'}),
#     html.Div([
#         html.Label("Please enter the second number"),
#         html.Br(),
#         html.Label("input"),
#         dcc.Input(id='input-b', type='number', value=1),
#     ], style={'margin-bottom': '20px'}),
#     #
#     html.Div([
#         # html.Label("Result:"),
#         html.Div(id='result'),
#     ], style={'margin-bottom': '20px'}),
#
# ])
#
# @my_app.callback(
#     Output('result', 'children'),
#     [Input('input-a', 'value'),
#      Input('input-b', 'value'),
#      Input('operation-dropdown', 'value')]
# )
# def calculate_result(a, b, operation):
#     try:
#         a = float(a)
#         b = float(b)
#     except ValueError:
#         return "Invalid input. Please enter valid numbers."
#
#     if operation == 'add':
#         result = a + b
#     elif operation == 'subtract':
#         result = a - b
#     elif operation == 'multiply':
#         result = a * b
#     elif operation == 'divide':
#         if b == 0:
#             return "Cannot divide by zero."
#         result = a / b
#     elif operation == 'log':
#         if a <= 0 or b <= 1:
#             return "Invalid input for logarithm."
#         result = math.log(a, b)
#     elif operation == 'root':
#         if (a < 0 and b%2==0) or b <= 0 or a==0 or not float(b).is_integer():
#             return "Invalid input for root operation."
#         result = a ** (1 / b)
#     else:
#         return "Invalid operation."
#
#     return f"The output value is {result}"
#
#
# # ================
# # Question 4
# # ===============
# question4_layout =html.Div([
#     html.H1("Polynomial Function Plotter"),
#
#     html.Div([
#         html.Label("Enter the order of the polynomial:"),
#         dcc.Input(id='input-order', type='number', value=2),
#     ], style={'margin-bottom': '20px'}),
#
#     dcc.Graph(id='polynomial-plot'),
# ])
# @my_app.callback(
#     Output('polynomial-plot', 'figure'),
#     [Input('input-order', 'value')]
# )
# def update_plot(order):
#     x_values = np.linspace(-2, 2, 1000)
#     y_values = x_values ** order
#
#     fig = px.line(x=x_values, y=y_values, labels={'x': 'x', 'y': f'f(x) = x^{order}'}, title='Polynomial Function Plot')
#     return fig
#
# # ================
# # Question 5
# # ===============
# question5_layout = html.Div([
#
#     html.Div([html.H1("Sinusoidal Function with Fast Fourier Transformation")],
#              style={'text-align': 'center'}),
#
#     html.Div([
#         html.Label("Please enter the number of sinusoidal cycle"),
#         html.Br(),
#         dcc.Input(id='cycles', type='number', value=5),
#         html.Br(),
#         html.Label("Please enter the mean of the white noise"),
#         html.Br(),
#         dcc.Input(id='mean', type='number', value=0),
#         html.Br(),
#         html.Label("Please enter the standard deviation of the white noise"),
#         html.Br(),
#         dcc.Input(id='std', type='number', value=0.1),
#         html.Br(),
#         html.Label("Please enter the number of samples"),
#         html.Br(),
#         dcc.Input(id='samples', type='number', value=1000),
#     ], style={'margin-bottom': '10px','width': '30%','text-justify': 'left'}),
#
#     html.Div([
#
#     ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'middle'}),
#     html.Div([dcc.Graph(id='sinusoidal-plot'),
#     dcc.Graph(id='fft-plot'),],style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'middle'})
# ])
#
# # Define callback to update the plots based on user input
# @my_app.callback(
#     [Output('sinusoidal-plot', 'figure'),
#      Output('fft-plot', 'figure')],
#     [Input('cycles', 'value'),
#      Input('mean', 'value'),
#      Input('std', 'value'),
#      Input('samples', 'value')]
# )
# def update_plots(cycles, mean, std, samples):
#     x = np.linspace(-np.pi, np.pi, samples)
#     noise = np.random.normal(mean, std, samples)
#     f_x = np.sin(cycles * x) + noise
#
#     # Plot f(x)
#     fig1 = go.Figure()
#     fig1.add_trace(go.Scatter(x=x, y=f_x, mode='lines', name='f(x)'))
#     fig1.update_layout(title='Sinusoidal Function with White Noise', xaxis_title='x', yaxis_title='y= f(x)')
#
#     # Plot FFT
#     f_x_fft = fft(f_x)
#     # freq = np.fft.fftfreq(samples, d=x[1] - x[0])
#     fig2 = go.Figure()
#     fig2.add_trace(go.Scatter(x=x, y=np.abs(f_x_fft), mode='lines', name='FFT'))
#     fig2.update_layout(title='The Fast Fourier Transform (FFT) of above generated data', xaxis_title='x', yaxis_title='y')
#
#     return fig1, fig2
#
# # ================
# # Question 6
# # ===============
# def logsig(x):
#     return 1 / (1 + np.exp(-x))
#
# def purelin(x):
#     return x
# def neural_network(b1_1, b1_2, w1_11, w1_21, b2_1, w2_11, w2_12, x):
#     a1_1 = logsig(w1_11 * x + b1_1)  # Log-sigmoid activation function
#     a1_2 = logsig(w1_21 * x + b1_2)
#     a2 = purelin((w2_11 * a1_1) + (w2_12 * a1_2) + b2_1)
#     return a2
#
# marks ={-10:"-10", -9: "-9", -8: "-8", -7: "-7", -6:"-6",
#                            -5:"-5",-4:"-4", -3:"-3",-2:"-2", -1:"-1", 0:"0",
#                            1:"1", 2:"2", 3: "3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"10"}
# question6_layout = html.Div([
#     html.H1("Two-Layered Neural Network"),
#
#     html.Div([
#         html.Img(src=my_app.get_asset_url('network_img.png')),
#         # html.Img(src='network_img.jpg'),
#         dcc.Graph(id='output-plot'),
#         html.Label("Adjust Parameters:"),
#         html.Br(),
#         # html.Label(r"$w_{1}ˆ{1}$ :"),
#
#         # dcc.Markdown(
#         # r""" $w_{1}$ """),
#         # html.Label(r"$b1_1$"),
#         dcc.Markdown('$b_{1}^1$', mathjax=True),
#         dcc.Slider(id='slider-b1_1', min=-10, max=10, step=0.001, value=0,
#                    marks=marks,
#                    ),
#         # html.Label(r"$b1_2$"),
#         dcc.Markdown('$b_{2}^1$', mathjax=True),
#         dcc.Slider(id='slider-b1_2', min=-10, max=10, step=0.001, value=0,
#                    marks=marks,
#                    ),
#         # html.Label(r"$w1_11$"),
#         dcc.Markdown('$w_{1,1}^1$', mathjax=True),
#         dcc.Slider(id='slider-w1_11', min=-10, max=10, step=0.001, value=1,
#                    marks =marks),
#         # html.Label(r"$w1_21$"),
#         dcc.Markdown('$w_{2,1}^1$', mathjax=True),
#         dcc.Slider(id='slider-w1_21', min=-10, max=10, step=0.001, value=1,
#                    marks =marks),
#         # html.Label(r"$b2_1$"),
#         dcc.Markdown('$b_1^2$', mathjax=True),
#         dcc.Slider(id='slider-b2_1', min=-10, max=10, step=0.001, value=0, marks=marks),
#         # html.Label(r"$w2_11$"),
#         dcc.Markdown('$w_{1,1}^2$', mathjax=True),
#         dcc.Slider(id='slider-w2_11', min=-10, max=10, step=0.001, value=1, marks=marks),
#         # html.Label(r"$w2_12$"),
#         dcc.Markdown('$w_{1,2}^2$', mathjax=True),
#         dcc.Slider(id='slider-w2_12', min=-10, max=10, step=0.001, value=1, marks=marks),
#
#     ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'middle'}),
# ])
#
# # Define callback to update the plot based on user input
# @my_app.callback(
#     Output('output-plot', 'figure'),
#     [Input('slider-b1_1', 'value'),
#      Input('slider-b1_2', 'value'),
# Input('slider-w1_11', 'value'),
# Input('slider-w1_21', 'value'),
#      Input('slider-b2_1', 'value'),
# Input('slider-w2_11', 'value'),
# Input('slider-w2_12', 'value'),
#
#    ]
# )
# def update_plot(b1_1, b1_2, w1_11, w1_21, b2_1, w2_11, w2_12):
#     x_values = np.linspace(-5, 5, 1000)
#     output_values = [neural_network(b1_1, b1_2, w1_11, w1_21, b2_1, w2_11, w2_12, x) for x in x_values]
#
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=x_values, y=output_values, mode='lines', name='Output (a2 vs p)'))
#     fig.update_layout(title='Two-Layered Neural Network Output', xaxis_title='p', yaxis_title='a2')
#     return fig
#
#
# #======================================================================================
# # ================
# # Parent call back
# # ===============
@my_app.callback(
Output(component_id='layout', component_property='children'),
Input(component_id='hw-questions', component_property='value')
)
def update_layout(ques):
    if ques=='q1':
        return question1_layout
    elif ques=='q2':
        return question2_layout
#     elif ques=='q3':
#         return question3_layout
#     elif ques=='q4':
#         return question4_layout
#     elif ques=='q5':
#         return question5_layout
#     elif ques=='q6':
#         return question6_layout
    else:
        return question1_layout
#
#
my_app.server.run(debug= True)