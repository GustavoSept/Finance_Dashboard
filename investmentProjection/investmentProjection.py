import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import pandas as pd

app = dash.Dash(__name__)

# External stylesheets to make it prettier
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(style={
    'fontFamily': 'Arial, sans-serif',
    'margin': '50px',
    'padding': '10px',
    'border': '1px solid #ddd',
    'borderRadius': '5px',
    'boxShadow': '2px 2px 20px #aaa'
}, children=[
    html.H1("Compound Interest Investment Calculator", style={
        'textAlign': 'center',
        'marginBottom': '30px'
    }),
    
    # Sliders & Input boxes
    html.Div(style={
        'marginBottom': '30px',
        'border': '1px solid #ccc',
        'padding': '20px',
        'borderRadius': '5px'
    }, children=[
        html.Div([
            html.Label("Investment Time (in years)"),
            dcc.Slider(
                id='investmentTime-slider',
                min=1,
                max=50,
                value=20,
                marks={i: '{}'.format(i) for i in range(0,51,5)},
                step=1
            )
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Label("Yield Rate (in annual %) "),
            dcc.Input(id='yieldRate-input', value=15, type='number')
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Label("Initial Contribution "),
            dcc.Input(id='initialContribution-input', value=0, type='number')
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Label("Monthly Contributions "),
            dcc.Input(id='monthlyContributions-input', value=500, type='number')
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Label("Yearly Productivity Gain (in %) "),
            dcc.Input(id='yearlyGainOnContributions-input', value=3, type='number')
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Label("Expected Inflation (in %) "),
            dcc.Input(id='expectedInflation-input', value=3.5, type='number')
        ], style={'marginBottom': '20px'}),
    ]),
    
    # Plot
    html.Div(style={
        'marginBottom': '30px',
        'padding': '20px'
    }, children=[
        dcc.Graph(id='compound-plot')
    ]),
    
    html.Div([
        html.H3(id='final-balance-display', children='', style={'textAlign': 'center'}),
        html.H3(id='comparison-display', children='', style={'textAlign': 'center'})
    ], style={'textAlign': 'center'})
])

def compound_interest_over_time(initialContribution,
                                monthlyContributions,
                                yieldRate,
                                investmentTime,
                                yearlyGainOnContributions,
                                expectedInflation):
    
        # Convert number from Human Readable to decimal
        yieldRate /= 100
        yearlyGainOnContributions /= 100
        expectedInflation /= 100
        
        # Convert annual rate to monthly
        monthly_rate = (1 + yieldRate) ** (1/12) - 1
        monthly_contrib_growth = (1 + yearlyGainOnContributions) ** (1/12) - 1
        monthly_inflation = (1 + expectedInflation) ** (1/12) - 1

        # ------ Initializing columns
        current_balance = [initialContribution]
        current_interest = [initialContribution * monthly_rate]
        current_inflation = [initialContribution * monthly_inflation]
        current_contribution = [monthlyContributions]
        final_balance = [current_balance[0] + current_interest[0] - current_inflation[0] + current_contribution[0]]

        investmentTime *= 12 # converting years to months    
        for month in range(1, investmentTime):
            # ------ Row by row calculation
            local_balance = final_balance[-1]
            local_interest = local_balance * monthly_rate
            local_inflation = local_balance * monthly_inflation
            local_contribution = monthlyContributions
            local_result = local_balance + local_interest - local_inflation + local_contribution

            # ------ Appending values to columns
            current_balance.append(local_balance)
            current_interest.append(local_interest)
            current_inflation.append(local_inflation)
            current_contribution.append(local_contribution)
            final_balance.append(local_result)

            # Increase the monthly contribution for the next month
            monthlyContributions *= (1 + monthly_contrib_growth)
        
        df = pd.DataFrame({            
            'Initial Balance': current_balance,
            'Interest': current_interest,
            'Inflation': current_inflation,
            'Monthly Investment': current_contribution,
            'Final Balance': final_balance
                            })
        df = df.round(2)
        
        df['Months'] = np.arange(1, investmentTime + 1)
        df['Current Year'] = df['Months'] // 12
        return df

def generate_scatter_trace(df, name, hover_name):
    return go.Scatter(
        x=df['Months'], 
        y=df['Final Balance'], 
        mode='lines', 
        name=name,
        text=df['Current Year'],
        hovertemplate=f'Month: %{{x}}<br>{hover_name}: $%{{y:,.2f}}<br>Current Year: %{{text}}'
    )


@app.callback(
    [Output('compound-plot', 'figure'),
     Output('final-balance-display', 'children'),
     Output('final-balance-display', 'style'),
     Output('comparison-display', 'children'),
     Output('comparison-display', 'style')],
    [
        Input('investmentTime-slider', 'value'),
        Input('yieldRate-input', 'value'),
        Input('initialContribution-input', 'value'),
        Input('monthlyContributions-input', 'value'),
        Input('yearlyGainOnContributions-input', 'value'),
        Input('expectedInflation-input', 'value')
    ]
)
def update_values(investmentTime, yieldRate, initialContribution, monthlyContributions, yearlyGainOnContributions, expectedInflation):
    # Calculate dataframes
    df = compound_interest_over_time(initialContribution,
                                     monthlyContributions,
                                     yieldRate,
                                     investmentTime,
                                     yearlyGainOnContributions,
                                     expectedInflation)
    df_Treasury = compound_interest_over_time(initialContribution,
                                     monthlyContributions,
                                     2,
                                     investmentTime,
                                     yearlyGainOnContributions,
                                     expectedInflation)
    df_stocks10 = compound_interest_over_time(initialContribution,
                                     monthlyContributions,
                                     12.39,
                                     investmentTime,
                                     yearlyGainOnContributions,
                                     expectedInflation)
    df_stocks20 = compound_interest_over_time(initialContribution,
                                     monthlyContributions,
                                     9.75,
                                     investmentTime,
                                     yearlyGainOnContributions,
                                     expectedInflation)
    
    # Create traces
    traces = [
        generate_scatter_trace(df, 'Your Investment', 'Final Balance'),
        generate_scatter_trace(df_Treasury, 'Average Treasury 3-month yield (2%)', 'Final Balance'),
        generate_scatter_trace(df_stocks10, 'Average Stock market yield, last 10 years (12.39%)', 'Final Balance'),
        generate_scatter_trace(df_stocks20, 'Average Stock market yield, last 20 years (9.75%)', 'Final Balance')
    ]
    
    layout = go.Layout(title="Compound Interest Over Time", xaxis=dict(title="Time in Months"), yaxis=dict(title="Amount"))
    
    # Display values
    final_balance = df['Final Balance'].iloc[-1]
    stock_balance = df_stocks10['Final Balance'].iloc[-1]
    difference = final_balance - stock_balance

    final_balance_content = f"Your Final Balance: ${final_balance:,.2f}"
    final_balance_style = {'fontSize': '24px', 'fontWeight': 'bold'}

    comparison_content = f"Difference compared to average stock market (last 10 years): ${difference:,.2f}"
    comparison_style = {
        'fontSize': '24px',
        'fontWeight': 'bold',
        'color': 'green' if difference >= 0 else 'red'
    }

    return {'data': traces, 'layout': layout}, final_balance_content, final_balance_style, comparison_content, comparison_style

if __name__ == '__main__':
    app.run_server(debug=True)