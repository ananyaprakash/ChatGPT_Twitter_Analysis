#########################
# Tab 3: User Based Analysis
##########################
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Sample data
df = pd.DataFrame({
    'username': ['user1', 'user2', 'user3', 'user4', 'user5'] * 20,
    'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'] * 20,
    'posts': [10, 5, 8, 12, 15] * 20
})

# Dash app
app = dash.Dash(__name__)

# Calculate the sum of observations by each unique value in 'username'
top_usernames = df['username'].value_counts().nlargest(10).index.tolist()

# Layout of the app
app.layout = html.Div(children=[
    html.H1(children='Dash App with Multiple Plots'),

    # Upper half: Horizontal bar plot
    dcc.Graph(
        id='bar-plot',
        figure=px.bar(df[df['username'].isin(top_usernames)], x='username', y='posts',
                      title='Top 10 Usernames by Sum of Observations', labels={'posts': 'Sum of Observations'})
    ),

    # Lower half: Pie chart and Area plot
    html.Div(children=[
        # Left side: Pie chart of segments based on 'username'
        dcc.Graph(
            id='pie-chart',
            figure=px.pie(df, names='username', title='Percentage of Total Observations by Username')
        ),

        # Right side: Area plot of posts segregated by 'month'
        dcc.Graph(
            id='area-plot',
            figure=px.area(df[df['username'].isin(top_usernames)], x='month', y='posts', color='username',
                           title='Area Plot of Posts Segregated by Month', labels={'posts': 'Number of Posts'})
        )
    ], style={'display': 'flex'})
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
