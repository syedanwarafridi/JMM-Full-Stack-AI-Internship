import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pandas as pd
import dash_daq as daq
import plotly.figure_factory as ff
import plotly.graph_objs as go
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix

data1 = datasets.load_iris()
df1 = pd.DataFrame(data1.data, columns=data1.feature_names)

data2 = datasets.load_digits()
df2 = pd.DataFrame(data2.data)

app = dash.Dash(__name__, external_stylesheets=['assets/styles.css'])

app.layout = html.Div(children=[ 
    html.H1('Dash Plotly Dashboard', style={'textAlign': 'center', 'font-weight': 'bold', 'color': 'black'}),
    
    html.Div([
        html.H2('Select a Dataset:', style={'textAlign': 'center', 'font-weight': 'bold', 'color': 'black'}),
        dcc.Dropdown(
            id='dataset-dropdown',
            style={'width': '50%', 'margin': '0 Auto'},
            options=[
                {'label': 'Iris Dataset', 'value': 'iris'},
                {'label': 'Digits Dataset', 'value': 'digits'}
            ],
            value='iris')
        ],
        className='form-group',  
    ),

    html.Div(id='table-container'),

    html.Div([
        html.H2('Select a Model:', style={'textAlign': 'center', 'font-weight': 'bold', 'color': 'black'}),
        dcc.Dropdown(
            id='model-dropdown',
            style={'width': '50%', 'margin': '0 Auto'},
            options=[
                {'label': 'Logistic Regression', 'value': 'logistic'},
                {'label': 'Decision Tree', 'value': 'decision_tree'},
                {'label': 'Random Forest', 'value': 'random_forest'}
            ],
            value='logistic')
        ],
        className='form-group',  
    ),

    html.Div([
        html.H2('Select a Metric:', style={'textAlign': 'center', 'font-weight': 'bold', 'color': 'black'}),
        dcc.Dropdown(
            id='metric-dropdown',
            style={'width': '50%', 'margin': '0 Auto'},
            options=[
                {'label': 'R2 Score', 'value': 'r2_score'},
                {'label': 'MAE (Mean Absolute Error)', 'value': 'mae'},
                {'label': 'MSE (Mean Squared Error)', 'value': 'mse'}
            ],
            value='r2_score')
        ],
        className='form-group',
    ),

    html.Div(
    children=[
        html.Button('Train Model', id='train-button', n_clicks=0, className='btn btn-primary',
            style={
                'fontSize': '18px', 
                'background-color': 'black',  
            }
        )
    ],
    style={
        'display': 'flex',
        'justify-content': 'center',
    }
),
    dcc.Textarea(id='training-results', style={'width': '100%', 'height': '200px'}),

    dcc.Graph(id='roc-graph') 
])

table_style = {
    'width': '60%',
    'margin': '20px auto',
}

@app.callback(
    Output('table-container', 'children'),
    Input('dataset-dropdown', 'value')
)
def update_table(selected_dataset):
    if selected_dataset == 'iris':
        df = df1
        columns = [{'name': col, 'id': col} for col in df.columns]
    else:
        df = df2
        columns = [{'name': str(col), 'id': str(col)} for col in df.columns]

    page_size = 5

    return dash_table.DataTable(
        id='selected-table',
        columns=columns,
        data=df.head(page_size).to_dict('records'),
        page_size=page_size,
        style_table=table_style,
        style_cell={'backgroundColor': 'black', 'color': 'white'}
    )

@app.callback(
    [Output('training-results', 'value'), Output('roc-graph', 'figure')],
    [Input('train-button', 'n_clicks')],
    [Input('model-dropdown', 'value'),
    Input('dataset-dropdown', 'value'),
    Input('metric-dropdown', 'value')]
)
def train_model(n_clicks, selected_model, selected_dataset, selected_metric):
    if n_clicks == 0:
        return "", go.Figure()

    if selected_model == 'logistic':
        model = LogisticRegression(max_iter=1000) 
    elif selected_model == 'decision_tree':
        model = DecisionTreeClassifier()
    elif selected_model == 'random_forest':
        model = RandomForestClassifier()
    else:
        return "No model selected.", go.Figure()

    if selected_dataset == 'iris':
        data = data1
    else:
        data = data2

    X = data.data
    y = data.target

    model.fit(X, y)

    if selected_metric == 'r2_score':
        metric_value = r2_score(y, model.predict(X))
    elif selected_metric == 'mae':
        metric_value = mean_absolute_error(y, model.predict(X))
    elif selected_metric == 'mse':
        metric_value = mean_squared_error(y, model.predict(X))
    else:
        return "Invalid metric selection.", go.Figure()

    cm = confusion_matrix(y, model.predict(X))

    cm_figure = ff.create_annotated_heatmap(
        z=cm,
        x=[str(label) for label in data.target_names],
        y=[str(label) for label in data.target_names],
        colorscale='Viridis',
    )
    cm_figure.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual')

    return f"{selected_metric}: {metric_value:.4f}", cm_figure

if __name__ == '__main__':
    app.run_server(debug=True)
