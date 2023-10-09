import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash import dash_table
from dash import no_update
from flask import Flask
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
server = dash_app.server

TOOLTIP_STYLE = {"background-color": "black", "color": "white", "border-radius": "5px"}
LABEL_STYLE = {'font-weight': 'bold'}
H6_STYLE = {'textAlign': 'center', 'padding': '5px', 'fontWeight': 'bold', 'fontStyle': 'italic'}

MAX_INVESTMENT_TIME = 20

dash_app.layout = dbc.Container([
    dbc.Row(
        dbc.Col([
            html.H1('Portfolio Value Projection', style={'textAlign': 'center', 'padding': '20px'}),
            
            # Investment Settings
            html.Div([
                html.H6('Investment Settings', style={**H6_STYLE, 'color': '#0a8a06'}),
                dbc.Row([
                    dbc.Col([
                        html.Label('Investment ID', style=LABEL_STYLE),
                        dcc.Input(id='investment-type', type='text', placeholder='Enter Investment ID', style={'width': '100%'}),
                        dbc.Tooltip("Enter the name or code of the investment",
                                    target="investment-type",
                                    style=TOOLTIP_STYLE,
                                    delay={"show": 450, "hide": 100},
                                    )
                    ], width=6, align="center"),
                    
                    dbc.Col([
                        html.Label('Investment Strategy', style=LABEL_STYLE),
                        dcc.Dropdown(id='risk-strategy', options=[
                            {'label': 'Conservative', 'value': 'conservative'},
                            {'label': 'Medium', 'value': 'medium'},
                            {'label': 'Risky', 'value': 'risky'}
                        ], value='risky'),
                        dbc.Tooltip("Set selling-point distance from ideal proportion. The riskier, the less often the asset is sold.",
                                    target="risk-strategy",
                                    style=TOOLTIP_STYLE,
                                    delay={"show": 450, "hide": 100},
                                    )
                    ], width=6, align="center")
                ]),

                html.Label('Ideal Proportion (%)', style=LABEL_STYLE),
                dcc.Slider(id='ideal-proportion-slider', min=0, max=100, step=1, value=50, 
                           marks={i: str(i) + "%" for i in range(0, 101, 10)}),
                dbc.Tooltip("Set the desired portfolio percentage for this investment. This slider can also be treated as weighted proportions.",
                            target="ideal-proportion-slider",
                            style=TOOLTIP_STYLE,
                            delay={"show": 450, "hide": 100},
                            ),
                
                dbc.Row([
                    dbc.Col([
                        html.Label('Expected Growth (%)', style=LABEL_STYLE),
                        dcc.Input(id='expected-growth', type='number', placeholder='Enter Expected Growth (%)', value=6, style={'width': '100%'}),
                        dbc.Tooltip("Input expected annual growth for the asset (in %).",
                                    target="expected-growth",
                                    style=TOOLTIP_STYLE,
                                    delay={"show": 450, "hide": 100},
                                    )
                    ], width=4, align="center"),

                    dbc.Col([
                        html.Label('Asset Volatility', style=LABEL_STYLE),
                        dcc.Dropdown(id='asset-volatility', options=[
                            {'label': 'Low', 'value': 'low'},
                            {'label': 'Mid', 'value': 'mid'},
                            {'label': 'High', 'value': 'high'}
                        ], disabled=True, value='high'),
                        dbc.Tooltip("Set how volatile the price of the asset is.",
                                        target="asset-volatility",
                                        style=TOOLTIP_STYLE,
                                        delay={"show": 450, "hide": 100},
                                        )                        
                    ], width=4, align="center"),

                    dbc.Col([
                        dcc.Checklist(id='growth-decay', options=[
                            {'label': 'Enable Growth Decay', 'value': 'True'}
                        ], value=[]),
                        dbc.Tooltip("If turned on, expected growth linearly decays close to the median growth.",
                                    target="growth-decay",
                                    style=TOOLTIP_STYLE,
                                    delay={"show": 450, "hide": 100},
                                    ),

                        dcc.Checklist(id='random-growth-check', options=[
                            {'label': 'Enable Random Growth', 'value': 'True'}
                        ], value=[]),
                        dbc.Tooltip("If turned on, volatility cycles are calculated for the asset.",
                                    target="random-growth-check",
                                    style=TOOLTIP_STYLE,
                                    delay={"show": 450, "hide": 100},
                                    )                        
                    ], width=4, align="center")
                    
                ]),
            ], style={'background': '#f5f5f5', 'padding': '2px 15px 15px 15px', 'borderRadius': '5px'}),
            
            html.Br(),

            # Advanced Investments Settings
            html.Div([
                html.H6('Advanced Investments Settings', style={**H6_STYLE, 'color': '#f26c0c'}),
                dbc.Row([
                    dbc.Col([
                        html.Label('Volatility Cycle Duration (in years)', style=LABEL_STYLE),
                        dcc.Slider(id='volatility-duration-slider', min=0.5, max=15, step=0.5, value=2, 
                                marks={i: str(i) for i in range(0, 16, 1)}),
                        dbc.Tooltip("Volatility Cycles keep the mean asset price intact, it just impacts the spread of the volatility.",
                                    target="volatility-duration-slider",
                                    style=TOOLTIP_STYLE,
                                    delay={"show": 450, "hide": 100},
                                    ),

                        html.Label('Volatility Magnitude Multiplier', style=LABEL_STYLE),
                        dcc.Input(
                            id='volatility-magnitude',
                            type='number',
                            placeholder='Enter Volatility Magnitude',
                            value=1,
                            style={'width': '100%'},
                            min = 0,
                            max = 5,
                            ),
                        dbc.Tooltip("Set how intense the volatility cycles are.",
                                    target="volatility-magnitude",
                                    style=TOOLTIP_STYLE,
                                    delay={"show": 450, "hide": 100},
                                    ),
                        html.Label('Volatility Phase', style=LABEL_STYLE),
                        dcc.Input(
                            id='volatility-phase',
                            type='number',
                            placeholder='Enter Volatility Phase',
                            value=0,
                            style={'width': '100%'},
                            min = 0,
                            max = 1,
                            ),
                        dbc.Tooltip("Set where the cycle should begin. 0 starts at peak stability. 1 starts at peak volatility.",
                                    target="volatility-phase",
                                    style=TOOLTIP_STYLE,
                                    delay={"show": 450, "hide": 100},
                                    )

                    ], width=6, align="center"),

                    dbc.Col([
                        html.Label('Bull-Bear Cycle Duration (in years)', style=LABEL_STYLE),
                        dcc.Slider(id='bullbear-duration-slider', min=0.5, max=15, step=0.5, value=5,
                                marks={i: str(i) for i in range(0, 16, 1)}),
                        dbc.Tooltip("Bull-Bear Cycles alter the mean price of the asset, up and down.",
                                    target="bullbear-duration-slider",
                                    style=TOOLTIP_STYLE,
                                    delay={"show": 450, "hide": 100},
                                    ),

                        html.Label('Bull-Bear Magnitude Multiplier', style=LABEL_STYLE),
                        dcc.Input(
                            id='bullbear-magnitude',
                            type='number',
                            placeholder='Enter Bull-Bear Magnitude',
                            value=1,
                            style={'width': '100%'},
                            min = 0,
                            max = 5,
                            ),
                        dbc.Tooltip("Set how intense the Bull-Bear cycles are.",
                                    target="bullbear-magnitude",
                                    style=TOOLTIP_STYLE,
                                    delay={"show": 450, "hide": 100},
                                    ),
                        html.Label('Bull-Bear Phase', style=LABEL_STYLE),
                        dcc.Input(
                            id='bullbear-phase',
                            type='number',
                            placeholder='Enter Bull-Bear Phase',
                            value=0,
                            style={'width': '100%'},
                            min = 0,
                            max = 1,
                            ),
                        dbc.Tooltip("Set where the cycle should begin. 0.25 starts at peak Bear. 0.75 starts at peak Bull.",
                                    target="bullbear-phase",
                                    style=TOOLTIP_STYLE,
                                    delay={"show": 450, "hide": 100},
                                    )
                    ], width=6, align="center")
                ])
            ], id='advanced-settings-div', style={'background': '#f5f5f5', 'padding': '2px 15px 15px 15px', 'borderRadius': '5px'}),
            
            html.Br(),

            # Portfolio Settings
            html.Div([
                html.H6('Portfolio Settings', style={**H6_STYLE, 'color': '#55bee0'}),
                dbc.Row([
                    dbc.Col([
                        html.Label('Investment Starting Point', style=LABEL_STYLE),
                        dcc.Input(id='investment-start-amount', type='number', placeholder='Enter Investment Amount', value=1000, style={'width': '100%'})
                    ], width=6, align="center"),
                    
                    dbc.Col([
                        html.Label('Monthly Investment', style=LABEL_STYLE),
                        dcc.Input(id='investment-monthly-amount', type='number', placeholder='Enter Investment Amount', value=100, style={'width': '100%'})
                    ], width=6, align="center")
                ]),
                
                html.Label('Investment Time (years)', style=LABEL_STYLE),
                dcc.Slider(id='investment-time-slider', min=0, max=MAX_INVESTMENT_TIME, step=1, value=2, 
                           marks={i: str(i) for i in range(0, MAX_INVESTMENT_TIME+1, 1)})
            ], style={'background': '#f5f5f5', 'padding': '2px 15px 15px 15px', 'borderRadius': '5px'}),

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
            html.Div(id='div-assetsBackup', style={'display': 'none'}),
            html.Div(id='hide-table-flag', style={'display': 'none'}),

            # Display Areas
            html.Div(id='table-div', style={'overflow': 'auto', 'width': '100%'}),
            dcc.Loading(
                id="loading-external",
                type="default",
                children=[
                    html.Div(id='charts-div', style={'height': '100%'})
                ],
                style={'height': '100%', 'display': 'flex', 'alignItems': 'flex-start'}
            ),
        ], style={'width': '50%', 'marginLeft': '20%', 'marginRight': '20%'})
    )  
], fluid=True, style={'marginTop': '20px'})

# --------------------- CALLBACKS SECTION --------------

# Storing values from the user, displaying table
@dash_app.callback(
    [
        Output('table-div', 'children'),
        Output('div-assetsBackup', 'children'),
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
        dash.dependencies.State('volatility-duration-slider', 'value'),
        dash.dependencies.State('volatility-magnitude', 'value'),
        dash.dependencies.State('volatility-phase', 'value'),
        dash.dependencies.State('bullbear-duration-slider', 'value'),
        dash.dependencies.State('bullbear-magnitude', 'value'),        
        dash.dependencies.State('bullbear-phase', 'value'),
        dash.dependencies.State('div-assetsBackup', 'children')
    ]
)
def update_investments_table(
    apply_n,clean_n,calc_n,investment_type,
    ideal_proportion,risk_strategy,investment_start_amount, investment_monthly_amount,
    investment_time,expected_growth,random_growth,
    asset_volatility,growth_decay,volatility_duration, volatility_magnitude, volatility_phase,
    bullbear_duration, bullbear_magnitude, bullbear_phase, prev_investments):

    global investments
    global portfolioSettings

    # ------------------------------------------------ ERROR CHECKING ----------------------------------
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
        return no_update, prev_investments, 'show', "Investment ID cannot be empty!"
    elif investment_type.upper() in [item['Investment ID'] for item in investments]:
        return no_update, prev_investments, 'show', "Investment ID can't be duplicated"

    # ------------------------------------------------ STORING VALUES ----------------------------------

    # Store investment values pertaining to the whole portfolio
    portfolioSettings['Investment Time (years)'] = min(investment_time, 40) # Just in case the front-end sends a huge value, cap at 40 years
    portfolioSettings['Start Investment Amount'] = investment_start_amount 
    portfolioSettings['Monthly Investment'] = investment_monthly_amount

    investment = {
        'Investment ID': investment_type.upper(),
        'Ideal Proportion (%)': ideal_proportion,
        'Investment Strategy': risk_strategy,        
        'Expected Growth (%)': expected_growth,
        'Random Growth': True if 'True' in random_growth else False,
        'Asset Volatility': asset_volatility,
        'Growth Decay': True if 'True' in growth_decay else False,
        'Volatility Duration': volatility_duration, 
        'Volatility Magnitude': volatility_magnitude,
        'Volatility Phase': volatility_phase,
        'BullBear Duration': bullbear_duration,
        'BullBear Magnitude': bullbear_magnitude,
        'BullBear Phase': bullbear_phase
    }

    investments.append(investment)

    # Convert investments list to DataFrame for display
    df = pd.DataFrame(investments)

    # Returns 4 things:
    # The table itself
    # Stores investments in the div-assetsBackup, for rollback reasons
    # 'show', to show the table again if it's hidden
    # remove error messages when adding investments (sending empty string)
    return dash_table.DataTable(
        id='investment-table',
        columns=[{'name': i, 'id': i} for i in df.columns],
        data=df.to_dict('records'),
        style_table={'width': '100%', 'overflowX': 'auto'}
    ), str(investments), 'show', ''

# ------------

def calc_portfolio(df, portfolioSettings):

    global investments

    if investments:
        df = pd.DataFrame(investments)
    else:
        return no_update
    
    
    startInvestment = portfolioSettings.get('Start Investment Amount', 0)
    monthlyInvestment = portfolioSettings.get('Monthly Investment', 0)
    investmentTime = portfolioSettings.get('Investment Time (years)', 0)


    distinctInvestments_amount = df.shape[0]
    # re-scaling idealProportion and expectedGrowth
    df['Ideal Proportion (%)'] /= df['Ideal Proportion (%)'].sum()
    df['Expected Growth (%)'] /= 100

    # Re-labeling risks and volatility
    df['Investment Strategy'] = df['Investment Strategy'].map({
                                                                'conservative': 1.0375,
                                                                'medium': 1.075,
                                                                'risky': 1.15
                                                            })

    df['Asset Volatility'] = df['Asset Volatility'].map({
                                                        'high': 10,
                                                        'mid': 6,
                                                        'low': 3
                                                    })
    
    # Disabling random number generation where necessary
    df.loc[df['Random Growth'] == False, 'Asset Volatility'] = 0

    # Basically a linear function, with sine-wave at the higher end to smooth it.
    thresholdProportion = \
        np.minimum(
            (np.sin(df['Ideal Proportion (%)'] * 0.5 * np.pi)
                + (df['Ideal Proportion (%)'] * 0.7))/ (0.7 + 1),
            df['Ideal Proportion (%)'] * df['Investment Strategy']
    )

    investmentTime_inWeeks = investmentTime * 52
    
    # Initializing balances and setting actual investment amount for each investment
    totalSold = np.zeros(distinctInvestments_amount)
    totalBought = np.zeros(distinctInvestments_amount)
    currentAmount = np.array(startInvestment * df['Ideal Proportion (%)'])

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

    # pre-calculate masks
    decayMask = df['Growth Decay'] == True

    # ------------ Pre calculating random values section
    def vectorized_genPseudoRdNum(
            weeks,randomStd, nameLen,
            vol_Dur,vol_Mag, volPhase,
            trend_Dur, trend_Mag, trendPhase
            ):
        
        '''
            Generates an ndArray of 'weeks' size with pre-calculations for bull-bear cycles, and volatility cycles.
            This function is divided in x steps:
                - It calculates a predictable seed
                - Transforms user input into formula-friendly versions
                - Use Sine wave functions to oscillate both cycles
                - Returns pseudo-random number for all weeks, with oscilating mean (bbCycles) and spread (volCycles)

        '''

        # This makes the seed predictable, enabling the user to test portfolio performance on average if he wants to.
        nameLen_ext = np.tile(nameLen, (len(weeks), 1)) # making nameLen match the shape of weeks
        seedCalc = ((weeks + nameLen_ext[:,0])).astype(int)
        np.random.seed(seedCalc)

        # Formula for translating years to the phased cycle 
        transformed_vol_Dur = (1/vol_Dur/52)
        transformed_trend_Dur = (1/trend_Dur/52)

        # Allows the user to pick the phase of each cycle
        vol_phaseShift = volPhase * np.pi / (transformed_trend_Dur * np.pi)
        bb_phaseShift = trendPhase * (2 * np.pi) / (transformed_trend_Dur * np.pi)
        
        volatilityCycle = np.maximum(
                            np.abs(
                                np.sin(
                                    (weeks[:, np.newaxis] - vol_phaseShift) * transformed_vol_Dur * np.pi
                                ) * vol_Mag
                            ) + 1e-10,
                        0.35) # Minimum Volatility allowed

        # ---------------- Making Volatility Cycles have a floor. Floor is smoothed for niceness
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        PARAM_FLOOR = 0.5
        scaling_factor = 1 / PARAM_FLOOR * 1.4 # empirically gives the smoothest curves for any PARAM_FLOOR
        scaled_volatility = (volatilityCycle - PARAM_FLOOR) * scaling_factor

        sVolatilityCycle = sigmoid(scaled_volatility) * (volatilityCycle - PARAM_FLOOR) + PARAM_FLOOR

        # wrapping trend's formula in a function allows me to easily cap de-growth
        def trendCycle_func(weeks, transformed_trend_Dur, bb_phaseShift, trend_Mag):
            return (np.sin((weeks[:, np.newaxis] - bb_phaseShift) * transformed_trend_Dur * np.pi) * trend_Mag)
        
        trendCycle = np.maximum(trendCycle_func(weeks, transformed_trend_Dur, bb_phaseShift, trend_Mag),
                        trendCycle_func(weeks, transformed_trend_Dur, bb_phaseShift, trend_Mag = np.minimum(0.95, trend_Mag)))


        # Create an array of random numbers with shape (investmentTime_inWeeks, number_of_rows)
        return np.array([np.random.normal(1 * (1+trendCycle[i]), randomStd * sVolatilityCycle[i]) 
                                for i in range(len(weeks))])

    if df['Random Growth'].any(): # skipping pre-calculation if there's no randomGrowth checked
        weeks = np.arange(1, investmentTime_inWeeks + 1)
        random_values = vectorized_genPseudoRdNum(
            weeks, 
            df['Asset Volatility'].to_numpy(), 
            df['Investment ID'].str.len().to_numpy(),
            df['Volatility Duration'].to_numpy(),
            df['Volatility Magnitude'].to_numpy(),
            df['Volatility Phase'].to_numpy(),
            df['BullBear Duration'].to_numpy(),
            df['BullBear Magnitude'].to_numpy(),
            df['BullBear Phase'].to_numpy()
        )

    results = []

    for week in range(1, investmentTime_inWeeks + 1):
        # Getting precalculated values for Decay Growth
        weekGrowthValues = decay_2DList[:, week-1]

        # Compound interest conversion from annual to weekly growth
        weekGrowth = (1 + weekGrowthValues) ** (1/52) - 1
        
        # skipping calculation if there's no randomGrowth checked
        if df['Random Growth'].any():
            weekGrowth *=  random_values[week - 1]


        # Casting compound growth
        currentAmount += currentAmount * weekGrowth
        
        # -------------------- Rebalancing Portfolio Section
        thresholdInvestment = thresholdProportion * currentAmount.sum()
        idealInvestment = df['Ideal Proportion (%)'] * currentAmount.sum()

        # Calculate the Selling Delta based on threshold trigger
        sellingDelta = np.maximum(currentAmount - thresholdInvestment, np.zeros(distinctInvestments_amount))        

        totalSold += sellingDelta

        # Update the 'Current Amount' column based on threshold trigger
        currentAmount = np.minimum(currentAmount, thresholdInvestment)
        soldAmount = sellingDelta.sum()

        # Calculate toBuy delta (how much each investment needs to be bought in theory)
        toBuy_Delta = np.maximum(idealInvestment - currentAmount, np.zeros(distinctInvestments_amount))

        # Actually 'buying' assets, with the money left in 'soldAmount' + Monthly Contributions
        #  First half: transform toBuy_Delta in proportion
        # Second half: multiply each proportion with the amount of money in the balance
        toBuy_Proportion = toBuy_Delta / (toBuy_Delta.sum() + 1e-10)
        boughtValues =  toBuy_Proportion * np.round(soldAmount + (monthlyInvestment*12/52), 2)

        totalBought += boughtValues
        actualProportion = currentAmount / (currentAmount.sum() + 1e-10)

        # --------------------------- Storing Info in TimeLine
        results.extend(list(zip(df['Investment ID'], currentAmount, [week]*distinctInvestments_amount, totalSold, totalBought, actualProportion)))


    return pd.DataFrame(
                        results,
                        columns=[
                            'Investment ID','Current Amount ($)','Week',
                            'Total Sold','Total Bought','Actual Proportion (%)'
                        ])

# Callback for disabling options when random-growth-check is off
@dash_app.callback(
    [
        Output('asset-volatility', 'disabled'),
        Output('advanced-settings-div', 'style')
    ],
    [Input('random-growth-check', 'value')]
)
def update_asset_volatility_and_advanced_settings(random_growth_value):
    is_disabled = len(random_growth_value) == 0
    # If checkbox is unchecked (i.e., value is empty), hide the div
    div_style = {'display': 'none'} if is_disabled else {'background': '#f5f5f5', 'padding': '2px 15px 15px 15px', 'borderRadius': '5px'}
    
    return is_disabled, div_style


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

    # Return early if there's no update
    if timeline_df is no_update:
        return ""


    # ------------------------------- calculations for plotting -------------------------------
    # Calculate the current worth of the portfolio
    current_worth = timeline_df[timeline_df['Week'] == timeline_df['Week'].max()]['Current Amount ($)'].sum()
    
    # Calculate the percentage growth compared to 'Start Investment Amount'
    percentage_growth = ((current_worth - investment_start_amount) / investment_start_amount) * 100

    # Converting Weeks to Years
    timeline_df['Current Year'] = timeline_df['Week'] // 52

    # ------------------------------- Plotting -------------------------------

    # Summary Info
    summary_div = html.Div([
        html.H3(f"Your portfolio is now worth: ${current_worth:,.2f}", style={'color': 'green', 'font-weight': 'bold'}),
        html.H5(f"Your portfolio grew by: {percentage_growth:.2f}%")
    ], style={'border': '1px solid #ddd', 'padding': '10px', 'border-radius': '5px', 'margin-bottom': '20px'})

    # Create the Pie Chart
    grouped_df = timeline_df.groupby('Investment ID').sum()['Actual Proportion (%)'].reset_index()
    pie_chart = dcc.Graph(
        figure=px.pie(
            grouped_df,
            names='Investment ID',
            values='Actual Proportion (%)',
            title="Investments Distribution"
            )
    )

    # Create the Line Chart for each Investment ID
    line_chart_by_type = dcc.Graph(
        figure=px.line(
            timeline_df,
            x='Week',
            y='Current Amount ($)',
            color='Investment ID',
            title="Current Amount ($) through Time by Investment ID",
            hover_data=['Current Year']
            )
    )

    # Create the Line Chart for the Total Amount
    priceHistory = timeline_df.groupby('Week', as_index=False)['Current Amount ($)'].sum()
    priceHistory['Current Year'] = priceHistory['Week'] // 52
    line_chart_total = dcc.Graph(
        figure=px.line(
            priceHistory,
            x='Week',
            y='Current Amount ($)',
            title="Total Amount ($) through Time",
            hover_data= ['Current Year']
            )
    )

    # Mini table with total sold and bought amounts for each asset
    mini_table_data = timeline_df[['Investment ID', 'Total Sold', 'Total Bought']].groupby('Investment ID', as_index = False).sum().round(2)
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

# Callback for hidding the table
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