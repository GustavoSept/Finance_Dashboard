# 📁 What's Inside This Repository?
This repo contains 2 dashboards, done with Python and Dash's plotly library:
## [**Portfolio Investment Projection**](https://github.com/GustavoSept/Finance_Dashboard/blob/main/dashboardApp.py)
It's the main finance dashboard. It works by receiving input from the user of each asset they own, and their properties. Then, it weekly calculates portfolio growth.

<img src="https://i.imgur.com/ggMyzaQ.jpg" alt="Main Dashboard settings" width="600" >
<img src="https://i.imgur.com/oyED7fW.jpg" alt="Main Dashboard summary statistics" width="600">
<img src="https://i.imgur.com/CNTGnRA.jpg" alt="Main Dashboard line charts" width="600">

- ***Asset properties:***
	- **Investment ID:** Name or class of the asset
	- **Ideal Proportion (%):** Sets the desired proportion that asset should occupy of the portfolio. The user can treat this as a weighted proportion, as the back-end treats all asset's Ideal Proportions to always sum up to 100%.
	- **Investment Strategy:** Sets how big the selling threshold should be, when asset value goes over the desired Ideal Proportion (%). User can set: Conservative, Medium and Risky. The riskier, the bigger the threshold before selling.
	- **Expected Growth:** Sets the mean annual growth of the asset.
	- **Growth Decay:** *[Optional]* If on, decays expected growth close to the median Growth over time.
	- **Random Growth:** *[Optional]* If on, oscillates the price of the asset, to simulate volatility and Bull-Bear cycles.
		- **Asset Volatility:** Sets the variance in asset price over time. User can set Low, Mid and High settings.
		- **Volatility Cycle Duration:** Sets the duration of volatility oscillation.
		- **Volatility Magnitude Multiplier:** Sets the strength of volatility cycles.
		- **Volatility Phase:** Lets the user offset where the volatility cycle will begin (on it's peak, valley, or anywhere in-between).
		- **Bull-Bear Cycle Duration:** Sets the duration of Bull-Bear oscillation.
		- **Bull-Bear Magnitude Multiplier:** Sets the strength of Bull-Bear cycles.
		- **Bull-Bear Phase:** Lets the user offset where the Bull-Bear cycle will begin (on it's peak, valley, or anywhere in-between).
- ***Asset properties:***
	- Investment Starting Point: How much money the user will have at week 0
	- Monthly Investment: How much money the user will invest every month
	- Investment Time: How many years the simulation will project
- ***Features:***
	- It accounts for compound growth and additional monthly investments.
	- Per-Asset Volatility and Bull-Bear cycles simulations
	- Simulates automated portfolio re-balancing week by week
	- Outputs:
		- Summary with what the portfolio is currently worth, and how much it grew in %
		- An interactive pie chart, with the proportion of each Investment ID
		- A table summarizing Total Amounts of Bought and Sold for each asset
		- An interactive line chart for each asset's worth over time
		- An interactive line chart with total portfolio worth over time
- ***Possible Improvements:***
	- GUI Overhaul
		- Make investment table editable
		- Add import/export capability to investment table (using .csv format, for instance)
	- Taxes
		- Add a way to configure profit tax payment, with possibility to set different tax brackets and exemption limits.
		- Display how many taxes was paid, from which investment
	- Inflation
		- Consider inflation metric to the calculation (decreasing Expected growth in each cycle)
		- Add a way for the user to dynamically set changes in expected inflation over time

## [**Simple Compound Yield Calculator:**](https://github.com/GustavoSept/Finance_Dashboard/tree/main/investmentProjection)
It's a simpler Dashboard that just projects annual Yield Rate into the future, and compares it with other common investments (Treasure and Stocks).

<img src="https://i.imgur.com/J0gGyfr.jpg" alt="Secondary Dashboard Layout" width="500">


