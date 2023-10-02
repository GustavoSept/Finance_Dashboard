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

import plotly.graph_objects as go
import plotly.express as px

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
                {'label': 'Mid', 'value': 'mid'},
                {'label': 'High', 'value': 'high'}
            ], disabled=True, value = 'high'),

            dcc.Checklist(id='growth-decay', options=[
                {'label': 'Enable Growth Decay', 'value': 'True'}
            ], value=[]),

            html.Br(),
            html.Button('Calculate Portfolio', id='calculate-button'),
            html.Button('Add Investment', id='apply-button', n_clicks=0), # Initializing n_clicks as well
            html.Button('Clean Table', id='clean-table-button', n_clicks=0),

            
            # Hidden div to store the previous state
            # This makes the Null check dynamic, for every new row 
            html.Div(id='hidden-div', style={'display': 'none'}),

            # Placeholder for results (either table or portfolio calculation)
            html.Div(id='table-div'),
            html.Div(id='charts-div')
        ], width={"size": 6, "offset": 3})),  # Adjusting width to 30% of screen and centering it
], fluid=True)

# --------------------- CALLBACKS SECTION --------------

# Storing values from the user, displaying table
@dash_app.callback(
    Output('table-div', 'children'),
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
        dash.dependencies.State('growth-decay', 'value'),
        dash.dependencies.State('hidden-div', 'children')
    ]
)
def update_investments_table(apply_n, clean_n, investment_type, ideal_proportion, risk_strategy, investment_amount, investment_time, expected_growth, random_growth, asset_volatility, growth_decay, prev_investments):
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
        'Asset Volatility': asset_volatility,
        'Growth Decay': True if 'True' in growth_decay else False

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

def calc_portfolio(df):
    if investments:
        df = pd.DataFrame(investments)
    else:
        return 'There seems to be no investments made yet'

    # re-scaling idealProportion and expectedGrowth
    df['Ideal Proportion (%)'] = df['Ideal Proportion (%)'] * (100 / df['Ideal Proportion (%)'].sum()) / 100
    df['Expected Growth (%)'] = df['Expected Growth (%)'] / 100

    # Re-labeling risks
    df['Risk Strategy'] = df['Risk Strategy'].replace({
                                                'conservative': 1.0375,
                                                'medium': 1.075,
                                                'risky': 1.15})
    
    # Basically a linear function, with sine-wave at the higher end to smooth it.
    df['thresholdProportion'] = \
        np.minimum(
            (np.sin(df['Ideal Proportion (%)'] * 0.5 * np.pi)
                + (df['Ideal Proportion (%)'] * 0.7))/ (0.7 + 1),
            df['Ideal Proportion (%)'] * df['Risk Strategy']
    )

    df['weeks'] = df['Investment Time (years)'] * 52
    df['Asset Volatility'] = df['Asset Volatility'].replace({'high':15.25, 'mid': 3.5, 'low':1})

    # Disabling random number generation where necessary
    df['Asset Volatility'] = df.apply(lambda row: 0 if row['Random Growth'] == 0 else row['Asset Volatility'], axis=1)
    
    # Initializing balances and setting actual investment amount for each investment
    df['Total Sold'] = np.zeros(df.shape[0])
    df['Total Bought'] = np.zeros(df.shape[0])
    df['Investment Amount ($)'] = df['Investment Amount ($)'] * df['Ideal Proportion (%)']
    investType = list(df['Investment Type'])
    currentAmount = list(df['Investment Amount ($)'])
    currentWeek = list(np.zeros(df.shape[0]))
    totalSold = list(np.zeros(df.shape[0]))
    totalBought = list(np.zeros(df.shape[0]))
    actualProportion = list(np.zeros(df.shape[0]))


    # (if enabled) Pre-calculate Expected Growth decay
    # Tends to the median growth (if growth > median)
    median_growth = df['Expected Growth (%)'].median()

    decay_2DList = np.array([
        np.linspace(
            start,
            ((median_growth * 10 + start) / 11),
            num=int(df['weeks'][0])
        ) if start > median_growth else np.full(int(df['weeks'][0]), start)
        for start in df['Expected Growth (%)']
    ])

    def genPseudoRdNum2(randomMean, randomStd, week):
        np.random.seed(week)
        volatilityCycle = np.abs(np.sin(week * 0.035 * np.pi) * 2) + 0.0001 # "Bull-Bear Market" every 2 years or so
        return np.random.normal(randomMean, randomStd * volatilityCycle)


    def genPseudoRdNum(randomMean, randomStd, week):
        np.random.seed(week)

        # About 2 Volatility cycles every year
        volatilityMagnitude = 0.5
        volatilityCycle \
            = (np.abs               
                (np.sin(week * 0.035 * np.pi) * volatilityMagnitude)             
            ) + 0.00001
        
        # Mini Bull-Bear cycles every 3 years
        trendMagnitude = 0.5
        trendCycle \
            = (np.sin(week * 0.006 * np.pi) * trendMagnitude) + 0.00001

        return np.random.normal(randomMean * (1+trendCycle), randomStd * volatilityCycle)


    for week in range(1, df['weeks'][0] + 1):
        # Decaying Growth from pre-calculated table
        mask = df['Growth Decay'] == True
        df.loc[mask, 'Expected Growth (%)'] = decay_2DList[mask, week-1]

        # Compound interest conversion from annual to weekly growth
        df['weeklyGrowth'] = ((1 + df['Expected Growth (%)']) ** (1/52) - 1)\
                                 * genPseudoRdNum(1, df['Asset Volatility'], week)
        # Calculating growth
        df['Investment Amount ($)'] += df['Investment Amount ($)'] * df['weeklyGrowth']
        
        balance = df['Investment Amount ($)'].sum() # new balance

        # -------------------- Rebalancing Portfolio Section
        
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

        # --------------------------- Storing Info in TimeLine
        investType.extend(df['Investment Type'].tolist())
        currentAmount.extend(df['Investment Amount ($)'].tolist())
        currentWeek.extend([week] * df.shape[0])
        totalSold.extend(df['Total Sold'].tolist())
        totalBought.extend(df['Total Bought'].tolist())
        actualProportion.extend(df['Actual Proportion (%)'].tolist()) 
        
        
    # Creating the actual dataFrame for timeline
    timeline_df = pd.DataFrame({
        'Investment Type': investType,
        'Current Amount ($)': currentAmount,
        'Week': currentWeek,
        'Total Sold': totalSold,
        'Total Bought': totalBought,
        'Actual Proportion (%)': actualProportion
    })
  

    return timeline_df

# Callback for enabling/disabling the assetVolatility dropdown based on randomGrowth checkbox
@dash_app.callback(
    [
        Output('asset-volatility', 'disabled'),
        Output('growth-decay', 'value'),
        Output('growth-decay', 'style')
    ],
    Input('random-growth-check', 'value')
)
def update_asset_volatility(random_growth_value):
    is_disabled = len(random_growth_value) == 0
    if is_disabled:
        # Return the default style for disabled look and set the value to False
        return is_disabled, [False], {'opacity': 0.5}
    else:
        # Return the normal style and set the value to True
        return is_disabled, [True], {'opacity': 1}





# Callback for plotting the calculation
@dash_app.callback(
    Output('charts-div', 'children'),
    Input('calculate-button', 'n_clicks')        
)
def calc_and_display_portfolio(n):
    global investments

    df = pd.DataFrame(investments)
    timeline_df = calc_portfolio(df)

    # Create the Pie Chart
    grouped_df = timeline_df.groupby('Investment Type').sum()['Actual Proportion (%)'].reset_index()
    pie_chart = dcc.Graph(
        figure=px.pie(
            grouped_df,
            names='Investment Type',
            values='Actual Proportion (%)',
            title="Investment Type Distribution"
            )
    )

    # Create the Line Chart for each Investment Type
    line_chart_by_type = dcc.Graph(
        figure=px.line(
            timeline_df,
            x='Week',
            y='Current Amount ($)',
            color='Investment Type',
            title="Current Amount ($) through Time by Investment Type"
            )
    )

    # Create the Line Chart for the Total Amount
    priceHistory = timeline_df.groupby('Week', as_index=False)['Current Amount ($)'].sum()
    line_chart_total = dcc.Graph(
        figure=px.line(
            priceHistory,
            x='Week',
            y='Current Amount ($)',
            title="Total Amount ($) through Time"
            )
    )

    return [pie_chart, line_chart_by_type, line_chart_total]



if __name__ == '__main__':
    investments = []  # Reset the investments list on server restart or page refresh
    app.run(debug=True)