import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from flask import Flask
import dash_table
from dash import no_update
import pandas as pd

# Global variable to store the investments
investments = []

app = Flask(__name__)
dash_app = dash.Dash(__name__, server=app, external_stylesheets=[dbc.themes.BOOTSTRAP], routes_pathname_prefix='/dash/')

dash_app.layout = dbc.Container([
    dbc.Row(
        dbc.Col([
            # Your Dash components for user input go here
            html.Label('Investment Type'),
            dcc.Input(id='investment-type', type='text', placeholder='Enter Investment Type'),
            html.Br(),

            html.Label('Ideal Proportion (%)'),
            dcc.Slider(id='ideal-proportion-slider', min=0, max=100, step=1, value=50, 
                       marks={i: str(i) + "%" for i in range(0, 101, 10)}),

            html.Label('Risk Strategy'),
            dcc.Dropdown(id='risk-strategy', options=[
                {'label': 'Conservative', 'value': 'conservative'},
                {'label': 'Medium', 'value': 'medium'},
                {'label': 'Risky', 'value': 'risky'}
            ], value='risky'),
            html.Br(),

            html.Label('Investment Amount ($)'),
            dcc.Input(id='investment-amount', type='number', placeholder='Enter Investment Amount'),
            html.Br(),

            html.Label('Investment Time (years)'),
            dcc.Slider(id='investment-time-slider', min=0, max=50, step=1, value=25, 
                       marks={i: str(i) for i in range(0, 51, 5)}),
            html.Br(),

            html.Label('Expected Growth (%)'),
            dcc.Input(id='expected-growth', type='number', placeholder='Enter Expected Growth (%)'),
            html.Br(),

            dcc.Checklist(id='random-growth-check', options=[
                {'label': 'Enable Random Growth', 'value': 'True'}
            ], value=[]),
            html.Br(),

            html.Label('Asset Volatility'),
            dcc.Dropdown(id='asset-volatility', options=[
                {'label': 'Low', 'value': 'low'},
                {'label': 'High', 'value': 'high'}
            ], disabled=True, value = 'high'),
            
            html.Br(),
            html.Button('Calculate Portfolio', id='calculate-button'),
            html.Button('Add Investment', id='apply-button', n_clicks=0), # Initializing n_clicks as well
            
            # Hidden div to store the previous state
            # This makes the Null check dynamic, for every new row 
            html.Div(id='hidden-div', style={'display': 'none'}),

            # Placeholder for results
            html.Div(id='output-div')
        ], width={"size": 6, "offset": 3})),  # Adjusting width to 30% of screen and centering it
], fluid=True)

# --------------------- CALLBACKS SECTION --------------

# Storing values from the user, displaying table
@dash_app.callback(
    Output('output-div', 'children'),
    Output('hidden-div', 'children'),
    [
        Input('apply-button', 'n_clicks')
    ],
    [
        dash.dependencies.State('investment-type', 'value'),
        dash.dependencies.State('ideal-proportion-slider', 'value'),
        dash.dependencies.State('risk-strategy', 'value'),
        dash.dependencies.State('investment-amount', 'value'),
        dash.dependencies.State('investment-time-slider', 'value'),
        dash.dependencies.State('expected-growth', 'value'),
        dash.dependencies.State('random-growth-check', 'value'),
        dash.dependencies.State('asset-volatility', 'value'),
        dash.dependencies.State('hidden-div', 'children')
    ]
)
def add_investment(n, investment_type, ideal_proportion, risk_strategy, investment_amount, investment_time, expected_growth, random_growth, asset_volatility, prev_investments):
    global investments

    # Check for None values
    inputs = [ideal_proportion, risk_strategy, investment_amount, expected_growth, asset_volatility]
    if any(val is None for val in inputs):
        # If we have previous investments stored, revert to them
        if prev_investments:
            investments = eval(prev_investments)
        return "Please fill out all fields before adding an investment!", prev_investments
    
    # Special Null and duplicated checks for investment_type, since it's a text value.
    if not investment_type or not investment_type.strip():
        return "Investment Type cannot be empty!", prev_investments
    elif investment_type in [item['Investment Type'] for item in investments]:
        return "Investment Type can't be duplicated", prev_investments

    # Apply InvestmentTime to all rows
    for investment in investments:
        investment['Investment Time (years)'] = investment_time
        investment['Investment Amount ($)'] = investment_amount    

    investment = {
        'Investment Type': investment_type,
        'Ideal Proportion (%)': ideal_proportion,
        'Risk Strategy': risk_strategy,
        'Investment Amount ($)': investment_amount,
        'Investment Time (years)': investment_time,
        'Expected Growth (%)': expected_growth,
        'Random Growth': True if 'True' in random_growth else False,
        'Asset Volatility': asset_volatility
    }
    investments.append(investment)

    # Convert investments list to DataFrame for display
    df = pd.DataFrame(investments)

    return dash_table.DataTable(
        id='investment-table',
        columns=[{'name': i, 'id': i} for i in df.columns],
        data=df.to_dict('records')
    ), str(investments)  # Storing the investments in the hidden-div, for rollback reasons

# ---------

# Callback for enabling/disabling the assetVolatility dropdown based on randomGrowth checkbox
@dash_app.callback(
    Output('asset-volatility', 'disabled'),
    [Input('random-growth-check', 'value')]
)
def update_asset_volatility(random_growth_value):
    return len(random_growth_value) == 0

if __name__ == '__main__':
    investments = []  # Reset the investments list on server restart or page refresh
    app.run(debug=True)
