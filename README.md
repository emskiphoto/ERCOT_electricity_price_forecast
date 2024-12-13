# ERCOT Electricity Price Forecast
__Project Objective:__
Generate hourly forecasts of the $/MWh Local Marginal Price (LMP) on the Day-Ahead (DA) and Real-Time (RT) markets.


Price forecasting will be generated for the "ERCOT North Hub" location, and data for various Independent System Operators(ISO) is considered.

__Workflow:__
The process is divided in to the following sequence of notebooks (.ipynb format):

 * 1 - ETL - DA & RT LMP Forecast
 * 2 - Descriptive Statistics & Data Augmentation
 * 3 - Stationarity and Seasonality
 * 4 - ARIMA Time Series Models - _coming soon_
 * 5 - ARCH_GARCH_models - _coming soon_

The script was originally written to handle a specific dataset.  However, it is built with flexibility that should permit the use of variable sets of time series and feature tabular data as long as they use the same structure as the original dataset.

__Source(s):__
(1) MS Excel file with two worksheets containing time-series data and tabular data.  

Limitations
* The script does not identify trading strategies or any other application of the forecast
* The forecast algorithms are trained on less than one year of data.

Copyright Matt Chmielewski<BR>
December 6, 2024<BR>
https://github.com/emskiphoto
