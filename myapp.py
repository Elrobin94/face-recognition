import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

st.write("""
# Stock Price App
Shown are the stock closing price and volume of Google!
""")

# Define the ticker symbol
tickerSymbol = 'GOOGL'

# Get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

# Get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start='2012-3-06', end='2024-3-08')

# Convert Timestamp to numeric representation
tickerDf['NumericDate'] = pd.to_numeric(tickerDf.index)

# Adding Moving Average as a feature
tickerDf['MA_50'] = tickerDf['Close'].rolling(window=50).mean()

# Drop NaN values
tickerDf.dropna(inplace=True)

# Convert index to UTC to remove timezone information
tickerDf.index = tickerDf.index.tz_localize(None)

# Open High Low Close Volume Dividends Stock Splits

st.write("""
## Closing Price
""")
st.line_chart(tickerDf.Close)

st.write("""
## Volume Price
""")
st.line_chart(tickerDf.Volume)

st.write("""
## Google Stock Data
""")
st.write(tickerDf)

# Prepare the data with additional features
X = tickerDf[['NumericDate', 'MA_50']]
y = tickerDf['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the root mean squared error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Convert NumericDate back to datetime for forecast_df
forecast_dates = pd.to_datetime(X_test['NumericDate'], unit='ns')

# Create forecast DataFrame with dates and predicted values
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Close': y_pred})

# Set timezone to UTC
forecast_df['Date'] = forecast_df['Date'].dt.tz_localize(None)

# Combine forecasted and actual data
combined_df = pd.concat([tickerDf[['Close']], forecast_df.set_index('Date')['Close']], axis=1)
combined_df.columns = ['Actual', 'Forecast']

st.write("""
## Stock Price Forecast
""")
st.line_chart(combined_df)
st.write(f"Root Mean Squared Error: {rmse}")