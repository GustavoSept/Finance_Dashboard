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
portfolioSettings = {
    'Start Investment Amount': 1000,
    'Monthly Investment': 100,
    'Investment Time (years)': 2
}


app = Flask(__name__)
dash_app = dash.Dash(__name__, server=app, external_stylesheets=[dbc.themes.BOOTSTRAP], routes_pathname_prefix='/dash/')

dash_app.layout = dbc.Container([
    dbc.Row(
        dbc.Col([
            html.H1('Portfolio Value Projection', style={'textAlign': 'center', 'padding': '20px'}),
            
            # Investment Settings
            html.Div([
                html.Label('Investment Type', style={'font-weight': 'bold'}),
                dcc.Input(id='investment-type', type='text', placeholder='Enter Investment Type', style={'width': '100%'}),
                
                html.Label('Ideal Proportion (%)', style={'font-weight': 'bold'}),
                dcc.Slider(id='ideal-proportion-slider', min=0, max=100, step=1, value=50, 
                           marks={i: str(i) + "%" for i in range(0, 101, 10)}),
                
                html.Label('Risk Strategy', style={'font-weight': 'bold'}),
                dcc.Dropdown(id='risk-strategy', options=[
                    {'label': 'Conservative', 'value': 'conservative'},
                    {'label': 'Medium', 'value': 'medium'},
                    {'label': 'Risky', 'value': 'risky'}
                ], value='risky'),

                html.Label('Expected Growth (%)', style={'font-weight': 'bold'}),
                dcc.Input(id='expected-growth', type='number', placeholder='Enter Expected Growth (%)', value=6, style={'width': '100%'}),
                
                dcc.Checklist(id='growth-decay', options=[
                    {'label': 'Enable Growth Decay', 'value': 'True'}
                ], value=[]),

                dcc.Checklist(id='random-growth-check', options=[
                    {'label': 'Enable Random Growth', 'value': 'True'}
                ], value=[]),
                
                html.Label('Asset Volatility', style={'font-weight': 'bold'}),
                dcc.Dropdown(id='asset-volatility', options=[
                    {'label': 'Low', 'value': 'low'},
                    {'label': 'Mid', 'value': 'mid'},
                    {'label': 'High', 'value': 'high'}
                ], disabled=True, value='high')
            ], style={'background': '#f7f7f7', 'padding': '15px', 'borderRadius': '5px'}),

            html.Br(),

            # Portfolio Settings
            html.Div([
                html.Label('Investment Starting Point', style={'font-weight': 'bold'}),
                dcc.Input(id='investment-start-amount', type='number', placeholder='Enter Investment Amount', value=1000, style={'width': '100%'}),
                
                html.Label('Monthly Investment', style={'font-weight': 'bold'}),
                dcc.Input(id='investment-monthly-amount', type='number', placeholder='Enter Investment Amount', value=100, style={'width': '100%'}),
                
                html.Label('Investment Time (years)', style={'font-weight': 'bold'}),
                dcc.Slider(id='investment-time-slider', min=0, max=40, step=1, value=2, 
                           marks={i: str(i) for i in range(0, 41, 5)})
            ], style={'background': '#e6e6e6', 'padding': '15px', 'borderRadius': '5px'}),

            html.Br(),

            # Action Buttons
            html.Div([
                html.Button('Calculate Portfolio', id='calculate-button', className='btn btn-primary', style={'marginRight': '10px'}),
                html.Button('Add Investment', id='apply-button', n_clicks=0, className='btn btn-secondary', style={'marginRight': '10px'}),
                html.Button('Clean Table', id='clean-table-button', n_clicks=0, className='btn btn-danger'),
            ]),

            # Error Message Area
            html.Div(id='error-message-div', style={'color': 'red', 'marginTop': '10px'}),

            # Hidden Containers
            html.Div(id='hidden-div', style={'display': 'none'}),
            html.Div(id='hide-table-flag', style={'display': 'none'}),

            # Display Areas
            html.Div(id='table-div'),
            html.Div(id='charts-div')
        ], width={"size": 6, "offset": 3})
    )  
], fluid=True, style={'marginTop': '20px'})

# --------------------- CALLBACKS SECTION --------------

# Storing values from the user, displaying table
@dash_app.callback(
    [
        Output('table-div', 'children'),
        Output('hidden-div', 'children'),
        Output('hide-table-flag', 'children'),
        Output('error-message-div', 'children')
    ]
    ,
    [
        Input('apply-button', 'n_clicks'),
        Input('clean-table-button', 'n_clicks'),
        Input('calculate-button', 'n_clicks')
    ],
    [
        dash.dependencies.State('investment-type', 'value'),
        dash.dependencies.State('ideal-proportion-slider', 'value'),
        dash.dependencies.State('risk-strategy', 'value'),
        dash.dependencies.State('investment-start-amount', 'value'),
        dash.dependencies.State('investment-monthly-amount', 'value'),
        dash.dependencies.State('investment-time-slider', 'value'),
        dash.dependencies.State('expected-growth', 'value'),
        dash.dependencies.State('random-growth-check', 'value'),
        dash.dependencies.State('asset-volatility', 'value'),
        dash.dependencies.State('growth-decay', 'value'),
        dash.dependencies.State('hidden-div', 'children')
    ]
)
def update_investments_table(
    apply_n,clean_n,calc_n,investment_type,
    ideal_proportion,risk_strategy,investment_start_amount, investment_monthly_amount,
    investment_time,expected_growth,random_growth,
    asset_volatility,growth_decay,prev_investments):

    global investments
    global portfolioSettings

    # Detecting which button was pressed
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'clean-table-button':
        investments = []
        return no_update, str(investments), 'hide', 'Table Cleaned'

    elif button_id == 'calculate-button':  # Check if calculate button was clicked
        # Check for investments data, if empty, return appropriate message
        if not investments:
            return no_update, no_update, 'show', 'Please add investments before calculating.'
        else:
            return no_update, no_update, 'hide', ''  # Hide the table and remove warnings

    # Check for None values
    inputs = [ideal_proportion, risk_strategy, investment_start_amount, investment_monthly_amount, expected_growth, asset_volatility]
    if any(val is None for val in inputs):
        # If we have previous investments stored, revert to them
        if prev_investments:
            investments = eval(prev_investments)
        return no_update, prev_investments, 'show', "Please fill out all fields before adding an investment!"

    
    # Special Null and duplicated checks for investment_type, since it's a text value.
    if not investment_type or not investment_type.strip():
        return no_update, prev_investments, 'show', "Investment Type cannot be empty!"
    elif investment_type.upper() in [item['Investment Type'] for item in investments]:
        return no_update, prev_investments, 'show', "Investment Type can't be duplicated"

    # Store investment values pertaining to the whole portfolio
    portfolioSettings['Investment Time (years)'] = min(investment_time, 40) # Just in case the front-end sends a huge value, cap at 40 years
    portfolioSettings['Start Investment Amount'] = investment_start_amount 
    portfolioSettings['Monthly Investment'] = investment_monthly_amount

    investment = {
        'Investment Type': investment_type.upper(),
        'Ideal Proportion (%)': ideal_proportion,
        'Risk Strategy': risk_strategy,        
        'Expected Growth (%)': expected_growth,
        'Random Growth': True if 'True' in random_growth else False,
        'Asset Volatility': asset_volatility,
        'Growth Decay': True if 'True' in growth_decay else False

    }

    investments.append(investment)

    # Convert investments list to DataFrame for display
    df = pd.DataFrame(investments)

    # Returns 4 things:
    # The table itself
    # Stores investments in the hidden-div, for rollback reasons
    # 'show', to show the table again if it's hidden
    # remove error messages when adding investments (sending empty string)
    return dash_table.DataTable(
        id='investment-table',
        columns=[{'name': i, 'id': i} for i in df.columns],
        data=df.to_dict('records')
    ), str(investments), 'show', ''
    
# ------------

def calc_portfolio(df, portfolioSettings):

    global investments

    if investments:
        df = pd.DataFrame(investments)
    
    
    startInvestment = portfolioSettings.get('Start Investment Amount', 0)
    monthlyInvestment = portfolioSettings.get('Monthly Investment', 0)
    investmentTime = portfolioSettings.get('Investment Time (years)', 0)


    distinctInvestments_amount = df.shape[0]
    # re-scaling idealProportion and expectedGrowth
    df['Ideal Proportion (%)'] = df['Ideal Proportion (%)'] * (100 / df['Ideal Proportion (%)'].sum()) / 100
    df['Expected Growth (%)'] = df['Expected Growth (%)'] / 100

    # Re-labeling risks
    df['Risk Strategy'] = df['Risk Strategy'].replace({
                                                'conservative': 1.0375,
                                                'medium': 1.075,
                                                'risky': 1.15})
    
    # Basically a linear function, with sine-wave at the higher end to smooth it.
    df['Treshold Proportion'] = \
        np.minimum(
            (np.sin(df['Ideal Proportion (%)'] * 0.5 * np.pi)
                + (df['Ideal Proportion (%)'] * 0.7))/ (0.7 + 1),
            df['Ideal Proportion (%)'] * df['Risk Strategy']
    )

    investmentTime_inWeeks = investmentTime * 52
    df['Asset Volatility'] = df['Asset Volatility'].replace({'high':15.25, 'mid': 3.5, 'low':1})

    # Disabling random number generation where necessary
    df['Asset Volatility'] = df.apply(lambda row: 0 if row['Random Growth'] == 0 else row['Asset Volatility'], axis=1)
    
    # Initializing balances and setting actual investment amount for each investment
    df['Total Sold'] = np.zeros(distinctInvestments_amount)
    df['Total Bought'] = np.zeros(distinctInvestments_amount)
    df['Current Amount'] = startInvestment * df['Ideal Proportion (%)']
    investType = list(df['Investment Type'])
    currentAmount = list(df['Current Amount'])
    currentWeek = list(np.zeros(distinctInvestments_amount))
    totalSold = list(np.zeros(distinctInvestments_amount))
    totalBought = list(np.zeros(distinctInvestments_amount))
    actualProportion = list(np.zeros(distinctInvestments_amount))


    # (if enabled) Pre-calculate Expected Growth decay
    # Tends to the median growth (if growth > median)
    median_growth = df['Expected Growth (%)'].median()

    decay_2DList = np.array([
        np.linspace(
            start,
            ((median_growth * 10 + start) / 11),
            num=investmentTime_inWeeks
        ) if start > median_growth else np.full(investmentTime_inWeeks, start)
        for start in df['Expected Growth (%)']
    ])

    # Decaying Growth from pre-calculated table
    decayMask = df['Growth Decay'] == True
    randomGrowthMask = df['Random Growth'] == True

    def genPseudoRdNum(randomMean, randomStd, week, rows, growthSum, nameLen):
        # This makes the seed predictable, enabling the user to test portfolio performance on average if he wants to.
        seedCalc = int((week + nameLen + rows) * growthSum)
        np.random.seed(seedCalc)

        # About 2 Volatility cycles every year
        volatilityMagnitude = 0.5
        volatilityCycle \
            = (np.abs               
                (np.sin(week * 0.035 * np.pi) * volatilityMagnitude)             
            ) + 0.00001
        
        # Mini Bull-Bear cycles every 3 years
        trendMagnitude = 0.5
        trendCycle \
            = (np.sin(week * 0.006 * np.pi) * trendMagnitude) - 0.00001

        return np.random.normal(randomMean * (1+trendCycle), randomStd * volatilityCycle)    

    for week in range(1, investmentTime_inWeeks + 1):
        
        # Getting precalculated values for Decay Growth (where True in decayMask)
        df.loc[decayMask, 'Expected Growth (%)'] = decay_2DList[decayMask, week-1]

        # Ccompound interest conversion from annual to weekly growth
        df['weeklyGrowth'] = (1 + df['Expected Growth (%)']) ** (1/52) - 1

        # Adding randomness only where randomGrowthMask is True
        df.loc[randomGrowthMask, 'weeklyGrowth'] = df.loc[randomGrowthMask, 'weeklyGrowth'] * genPseudoRdNum(1,
                                                                                                             df.loc[randomGrowthMask,
                                                                                                                    'Asset Volatility'],
                                                                                                                    week,
                                                                                                                    rows = df.shape[0],
                                                                                                                    growthSum = df['Expected Growth (%)'].sum(),
                                                                                                                    nameLen = df['Investment Type'].str.len().sum()
                                                                                                            )

        # Casting compound growth
        df['Current Amount'] += df['Current Amount'] * df['weeklyGrowth']        
        
        balance = df['Current Amount'].sum() # new balance

        # -------------------- Rebalancing Portfolio Section
        
        df['Threshold Investment Amount'] = df['Treshold Proportion'] * balance
        df['Ideal Investment Amount'] = df['Ideal Proportion (%)'] * balance

        # Calculate the Selling Delta based on threshold trigger
        df['Selling Delta'] = np.where(df['Current Amount'] > df['Threshold Investment Amount'],
                                    df['Current Amount'] - df['Threshold Investment Amount'],
                                    0)
        df['Total Sold'] += df['Selling Delta']

        # Update the 'Current Amount' column based on threshold trigger
        df.loc[df['Current Amount'] > df['Threshold Investment Amount'],
               'Current Amount'] = df['Threshold Investment Amount']
        soldAmount = df['Selling Delta'].sum()

        oldValues_Series = df['Current Amount'] # Later, we'll calculate how much we bought

        # Calculate toBuy delta (how much each investment needs to be bought in theory)
        df['toBuy Delta'] = np.where(df['Current Amount'] < df['Ideal Investment Amount'],
                                    df['Ideal Investment Amount'] - df['Current Amount'],
                                    0)
        # Actually 'buying' assets, with the money left in 'soldAmount' + Monthly Contributions
        df['Current Amount'] = np.where(df['Current Amount'] < df['Ideal Investment Amount'],
                                            ((df['toBuy Delta'] * (100 / df['toBuy Delta'].sum()) / 100)
                                                * int(soldAmount + monthlyInvestment)) + df['Current Amount'],
                                            df['Current Amount']
                                        )


        df['Total Bought'] = df['Total Bought'] + (df['Current Amount'] - oldValues_Series)
        df['Actual Proportion (%)'] = df['Current Amount'] / df['Current Amount'].sum()

        # --------------------------- Storing Info in TimeLine
        investType.extend(df['Investment Type'].tolist())
        currentAmount.extend(df['Current Amount'].tolist())
        currentWeek.extend([week] * distinctInvestments_amount)
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
    Output('asset-volatility', 'disabled'),
    Input('random-growth-check', 'value')
)
def update_asset_volatility(random_growth_value):
    return len(random_growth_value) == 0

# Callback for plotting the calculation
@dash_app.callback(
    Output('charts-div', 'children'),
    Input('calculate-button', 'n_clicks'),
    [
        dash.dependencies.State('investment-start-amount', 'value'),
        dash.dependencies.State('investment-monthly-amount', 'value'),
        dash.dependencies.State('investment-time-slider', 'value')
    ]      
)
def calc_and_display_portfolio(n, investment_start_amount, investment_monthly_amount, investment_time):
    global investments

    # Updating the global portfolioSettings before calling calc_portfolio
    portfolioSettings['Investment Time (years)'] = min(investment_time, 40) # Just in case the front-end sends a huge value, cap at 40 years
    portfolioSettings['Start Investment Amount'] = investment_start_amount 
    portfolioSettings['Monthly Investment'] = investment_monthly_amount

    df = pd.DataFrame(investments)
    timeline_df = calc_portfolio(df, portfolioSettings)

    # When there's no data in 'investments', calc_portfolio returns a string
    # So if there's no data, don't bother plotting
    if isinstance(timeline_df, str):
        return timeline_df


     # Calculate the current worth of the portfolio
    current_worth = timeline_df[timeline_df['Week'] == timeline_df['Week'].max()]['Current Amount ($)'].sum()
    
    # Calculate the percentage growth compared to 'Start Investment Amount'
    percentage_growth = ((current_worth - investment_start_amount) / investment_start_amount) * 100

    # Summary Info
    summary_div = html.Div([
        html.H3(f"Your portfolio is now worth: ${current_worth:,.2f}", style={'color': 'green', 'font-weight': 'bold'}),
        html.H5(f"Your portfolio grew by: {percentage_growth:.2f}%")
    ], style={'border': '1px solid #ddd', 'padding': '10px', 'border-radius': '5px', 'margin-bottom': '20px'})

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

    # Mini table with total sold and bought amounts for each asset
    mini_table_data = timeline_df[['Investment Type', 'Total Sold', 'Total Bought']].groupby('Investment Type', as_index = False).sum().round(2)
    mini_table = dash_table.DataTable(
        id='mini-investment-table',
        columns=[{'name': i, 'id': i} for i in mini_table_data.columns],
        data=mini_table_data.to_dict('records'),
        style_table={'margin-top': '20px'}
    )

    return [
        summary_div,        
        # Row for the charts
        dbc.Row([
            dbc.Col([
                mini_table
            ], width=6, align="center"),
            
            dbc.Col([
                pie_chart
            ], width=6)
        ]),
        line_chart_by_type, 
        line_chart_total
    ]


@dash_app.callback(
    Output('table-div', 'style'),
    Input('hide-table-flag', 'children')
)
def toggle_table_display(flag):
    if flag == 'hide':
        return {'display': 'none'}  # Hide the table
    elif flag == 'show':
        return {}  # Show the table
    else:
        return {}  # Default state (show the table)




if __name__ == '__main__':
    investments = []  # Reset the investments list on server restart or page refresh
    app.run(debug=True)