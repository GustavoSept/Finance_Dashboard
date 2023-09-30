import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import pandas as pd

app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Compound Interest Calculator"),
    
    # Sliders & Input boxes
    html.Label("Investment Time (in years)"),
    dcc.Slider(
        id='investmentTime-slider',
        min=1,
        max=50,
        value=20,
        marks={i: '{}'.format(i) for i in range(0,51,5)},
        step=1
    ),
    html.Div([
        html.Label("Yield Rate (in annual %)"),
        dcc.Input(id='yieldRate-input', value=15, type='number')
    ]),
    html.Div([
        html.Label("Initial Contribution"),
        dcc.Input(id='initialContribution-input', value=0, type='number')
    ]),
    html.Div([
        html.Label("Monthly Contributions"),
        dcc.Input(id='monthlyContributions-input', value=500, type='number')
    ]),
    html.Div([
        html.Label("Yearly Productivity Gain (in %)"),
        dcc.Input(id='yearlyGainOnContributions-input', value=3, type='number')
    ]),
    html.Div([
        html.Label("Expected Inflation (in %)"),
        dcc.Input(id='expectedInflation-input', value=3.5, type='number')
    ]),
    
    # Plot
    dcc.Graph(id='compound-plot')
])

# Callback function to update the plot
@app.callback(
    Output('compound-plot', 'figure'),
    [
        Input('investmentTime-slider', 'value'),
        Input('yieldRate-input', 'value'),
        Input('initialContribution-input', 'value'),
        Input('monthlyContributions-input', 'value'),
        Input('yearlyGainOnContributions-input', 'value'),
        Input('expectedInflation-input', 'value')
    ]
)
def update_plot(investmentTime, yieldRate, initialContribution, monthlyContributions, yearlyGainOnContributions, expectedInflation):
    
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

        months = np.arange(1, investmentTime + 1)    
        yearsList = months // 12

        return pd.DataFrame({
            'Current Year': yearsList,
            'Months': months,
            'Initial Balance': current_balance,
            'Interest': current_interest,
            'Inflation': current_inflation,
            'Monthly Investment': current_contribution,
            'Final Balance': final_balance
                            })

    
    df = compound_interest_over_time(initialContribution,
                                     monthlyContributions,
                                     yieldRate,
                                     investmentTime,
                                     yearlyGainOnContributions,
                                     expectedInflation)
    df_Treasury = compound_interest_over_time(initialContribution,
                                     monthlyContributions,
                                     yieldRate = 2,
                                     investmentTime=investmentTime,
                                     yearlyGainOnContributions=yearlyGainOnContributions,
                                     expectedInflation=expectedInflation)
    df_stocks10 = compound_interest_over_time(initialContribution,
                                     monthlyContributions,
                                     yieldRate = 12.39,
                                     investmentTime=investmentTime,
                                     yearlyGainOnContributions=yearlyGainOnContributions,
                                     expectedInflation=expectedInflation)
    df_stocks20 = compound_interest_over_time(initialContribution,
                                     monthlyContributions,
                                     yieldRate = 9.75,
                                     investmentTime=investmentTime,
                                     yearlyGainOnContributions=yearlyGainOnContributions,
                                     expectedInflation=expectedInflation)
    
    trace = go.Scatter(x=df['Months'], 
                       y=df['Final Balance'], 
                       mode='lines', 
                       name='Your Investment Return',
                       text=df['Current Year'],
                       hovertemplate='Month: %{x}<br>Final Balance: %{y}<br>Current Year: %{text}')
    trace2 = go.Scatter(x=df_Treasury['Months'], 
                       y=df_Treasury['Final Balance'], 
                       mode='lines', 
                       name='Average Treasury 3-month yield (2%)',
                       text=df_Treasury['Current Year'],
                       hovertemplate='Month: %{x}<br>Final Balance: %{y}<br>Current Year: %{text}')
    trace3 = go.Scatter(x=df_stocks10['Months'], 
                       y=df_stocks10['Final Balance'], 
                       mode='lines', 
                       name='Average Stock market yield, last 10 years (12.39%)',
                       text=df_stocks10['Current Year'],
                       hovertemplate='Month: %{x}<br>Final Balance: %{y}<br>Current Year: %{text}')
    trace4 = go.Scatter(x=df_stocks20['Months'], 
                       y=df_stocks20['Final Balance'], 
                       mode='lines', 
                       name='Average Stock market yield, last 20 years (9.75%)',
                       text=df_stocks20['Current Year'],
                       hovertemplate='Month: %{x}<br>Final Balance: %{y}<br>Current Year: %{text}')
        
    layout = go.Layout(title="Compound Interest Over Time", xaxis=dict(title="Time in Months"), yaxis=dict(title="Amount"))
    
    return {'data': [trace, trace2, trace3, trace4], 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True)