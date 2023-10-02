import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from flask import Flask
from dash import dash_table
from dash import no_update
import pandas as pd
import numpy as np

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
            dcc.Input(id='investment-amount', type='number', placeholder='Enter Investment Amount',value=500), 
            html.Br(),

            html.Label('Investment Time (years)'),
            dcc.Slider(id='investment-time-slider', min=0, max=50, step=1, value=25, 
                       marks={i: str(i) for i in range(0, 51, 5)}),
            html.Br(),

            html.Label('Expected Growth (%)'),
            dcc.Input(id='expected-growth', type='number', placeholder='Enter Expected Growth (%)',value=6),
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
            html.Button('Clean Table', id='clean-table-button', n_clicks=0),

            
            # Hidden div to store the previous state
            # This makes the Null check dynamic, for every new row 
            html.Div(id='hidden-div', style={'display': 'none'}),

            # Placeholder for results (either table or portfolio calculation)
            html.Div(id='output-div')
        ], width={"size": 6, "offset": 3})),  # Adjusting width to 30% of screen and centering it
], fluid=True)

# --------------------- CALLBACKS SECTION --------------

# Storing values from the user, displaying table
@dash_app.callback(
    Output('output-div', 'children'),
    Output('hidden-div', 'children'),
    [
        Input('apply-button', 'n_clicks'),
        Input('clean-table-button', 'n_clicks')
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
def update_investments_table(apply_n, clean_n, investment_type, ideal_proportion, risk_strategy, investment_amount, investment_time, expected_growth, random_growth, asset_volatility, prev_investments):
    global investments

    # Detecting which button was pressed
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'clean-table-button':
        investments = []
        return 'Table Cleaned', str(investments)

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
    elif investment_type.upper() in [item['Investment Type'] for item in investments]:
        return "Investment Type can't be duplicated", prev_investments

    # Apply investment_time and investment_amount to all rows
    for investment in investments:
        investment['Investment Time (years)'] = investment_time
        investment['Investment Amount ($)'] = investment_amount    

    investment = {
        'Investment Type': investment_type.upper(),
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


# Callback for doing the actual calculation
@dash_app.callback(
    Output('output-div', 'children'),
    Input('calculate-button', 'n_clicks')
        
)
def calc_portfolio(n):
    global investments

    df = pd.DataFrame(investments)

    # re-scaling idealProportion and expectedGrowth
    df['Ideal Proportion (%)'] = df['Ideal Proportion (%)'] * (100 / df['Ideal Proportion (%)'].sum()) / 100
    df['Expected Growth (%)'] = df['Expected Growth (%)'] / 100

    # Calculating thresholdProportion
    df['Risk Strategy'] = df['Risk Strategy'].replace({
                                                'Conservative': 7,
                                                'Medium': 2,
                                                'Risky': 1})
    
    # Simplified function to produce a 'slightly straight' sine curve.
    # The higher the value in Risk Strategy, the closer to a 1:1 mapping it is.
    # Zero being a 'pure' sine wave.
    df['thresholdProportion'] = \
        np.minimum(
            (np.sin(df['Ideal Proportion (%)'] * 0.5 * np.pi)
                + (df['Ideal Proportion (%)'] * df['Risk Strategy']))/ (df['Risk Strategy'] + 1),
            df['Ideal Proportion (%)'] + 0.01
    )

    df['weeks'] = df['Investment Time (years)'] * 52
    df['Asset Volatility'] = df['Asset Volatility'].replace({'High':2.65, 'Mid': 0.95, 'Low':0.25})

    # Disabling random number generation where necessary
    df['Asset Volatility'] = df.apply(lambda row: 0 if row['Random Growth'] == 0 else row['Asset Volatility'], axis=1)
    
    # Initializing balances and setting actual investment amount for each investment
    balance = df['Investment Amount ($)']
    df['Total Sold'] = np.zeros(df.shape[0])
    df['Total Bought'] = np.zeros(df.shape[0])
    df['Investment Amount ($)'] = df['Investment Amount ($)'] * df['Ideal Proportion (%)']

    def genPseudoRdNum(randomMean, randomStd, seed = 42):
        np.random.seed(seed)
        return np.random.normal(randomMean, randomStd)

    for week in range(df['weeks'][0]):
        # Compound interest conversion from annual to weekly growth
        df['weeklyGrowth'] = ((1 + df['Expected Growth (%)']) ** (1/52) - 1)\
                                 * genPseudoRdNum(1, df['Asset Volatility']/1.2, week)
        # Calculating growth
        df['Investment Amount ($)'] += df['Investment Amount ($)'] * df['weeklyGrowth']
        
        
        balance = df['Investment Amount ($)'].sum() # new balance
        # ------------ Rebalancing Portfolio
        
        df['Threshold Investment Amount'] = df['thresholdProportion'] * balance
        df['Ideal Investment Amount'] = df['Ideal Proportion (%)'] * balance

        # Calculate the Selling Delta based on threshold trigger
        df['Selling Delta'] = np.where(df['Investment Amount ($)'] > df['Threshold Investment Amount'],
                                    df['Investment Amount ($)'] - df['Threshold Investment Amount'],
                                    0)
        df['Total Sold'] += df['Selling Delta']

        #display('Before',df) # --------------------------------- DISPLAY

        # Update the 'Investment Amount ($)' column based on threshold trigger
        df.loc[df['Investment Amount ($)'] > df['Threshold Investment Amount'],
               'Investment Amount ($)'] = df['Threshold Investment Amount']
        soldAmount = df['Selling Delta'].sum()

        oldValues_Series = df['Investment Amount ($)'] # Later, we'll calculate how much we bought

        # Calculate toBuy delta (how much each investment needs to be bought in theory)
        df['toBuy Delta'] = np.where(df['Investment Amount ($)'] < df['Ideal Investment Amount'],
                                    df['Ideal Investment Amount'] - df['Investment Amount ($)'],
                                    0)
        # Actually 'buying' assets, with the money left in 'soldAmount'
        df['Investment Amount ($)'] = np.where(df['Investment Amount ($)'] < df['Ideal Investment Amount'],
                                            ((df['toBuy Delta'] * (100 / df['toBuy Delta'].sum()) / 100)
                                                * soldAmount) + df['Investment Amount ($)'],
                                            df['Investment Amount ($)']
                                        )
        df['Total Bought'] = df['Total Bought'] + (df['Investment Amount ($)'] - oldValues_Series)
        df['Actual Proportion (%)'] = df['Investment Amount ($)'] / df['Investment Amount ($)'].sum()
    

    return 'function not completed yet'



if __name__ == '__main__':
    investments = []  # Reset the investments list on server restart or page refresh
    app.run(debug=True)