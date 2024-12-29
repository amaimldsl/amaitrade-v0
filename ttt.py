import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import statistics
import yfinance as yf
import seaborn as sns
from pandas_datareader import data
from pulp import *
from  pypfopt import expected_returns, EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# Define the ticker symbol
#ticker = 'CL=F'  # Example: Crude Oil Futures
ticker = "SPUS"

# Get the data for the given ticker
data = yf.Ticker(ticker)

# Fetch the latest market data
latest_data = data.history(period="5d")
# Check if data exists before trying to access the last row
if not latest_data.empty:
    latest_price = latest_data['Close'].iloc[-1]
    print(f"The latest price of {ticker} is: {latest_price}")
else:
    print(f"No price data found for {ticker}.")