#########################
# Tab 2: Outlier analysis
##########################

from dash import Dash
from dash import html, dcc, Output, Input
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import date
import plotly.graph_objects as go


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
url = 'https://raw.githubusercontent.com/ananyaprakash/ChatGPT_Twitter_Analysis/main/ChatGPT_tweets_processed_50k.csv'
df= pd.read_csv(url)
df.reset_index(inplace=True)
# df_red = df.iloc[50000:]
# df_red.to_csv('ChatGPT_tweets_processed_50k.csv', index=False)
# print(df.head(10))
features= ['like_count' , 'retweet_count',  'content_length','hashtag_count','tagged_users_count']
#%%
print(df.shape)
df_date_grouped = df.groupby(['date'])['date'].count()
print(df_date_grouped.head(10))
#%%


# Dash app
my_app =  Dash('My app', external_stylesheets= external_stylesheets)

# Layout of the app
my_app.layout = html.Div(children=[
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
            height=800  # Set your desired height
        )
        return fig
    elif selected_plot_type == 'violin':
        fig = px.violin(df, y=selected_features)
        fig.update_layout(
            height=800  # Set your desired height
        )
        return fig





# Run the app
if __name__ == '__main__':
    my_app.run_server(debug=True)
