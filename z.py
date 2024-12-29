import yfinance as yf
import pandas as pd
import datetime
from pypfopt import expected_returns

def main():
    islamic_us_etfs = ['SPUS', 'HLAL', 'SPSK', 'SPRE', 'SPTE', 'SPWO', 'UMMA']
    market_index = "^GSPC"  # Use S&P 500 as the market benchmark

    start_date = "2020-01-01"
    end_date = "2024-12-26"

    # Download data for the Islamic US ETFs and market index
    assets_historical_data = yf.download(islamic_us_etfs, start=start_date, end=end_date, actions=True)
    market_historical_data = yf.download(market_index, start=start_date, end=end_date, actions=True)

    # Adjusted close for the assets (already calculated in your code)
    assets_adjusted_close = get_adjusted_close(islamic_us_etfs, assets_historical_data, start_date, end_date)

    # Adjusted close for the market (same process as assets)
    market_adjusted_close = get_adjusted_close([market_index], market_historical_data, start_date, end_date)

    
    # Check the adjusted close data
    check_data(assets_adjusted_close)
    check_data(market_adjusted_close)
    
    
    # Use adjusted close for the assets and market index
    retornos1 = expected_returns.capm_return(assets_adjusted_close, market_prices=market_adjusted_close, 
                                             returns_data=True, risk_free_rate=.04, frequency=252)

    print(retornos1)

def get_adjusted_close(assets, assets_historical_data, start_date, end_date):
    assets_adjusted_close = pd.DataFrame()
    assets_close_minus_dividends = assets_historical_data["Close"] - assets_historical_data["Dividends"]
    assets_adjusted_close = assets_close_minus_dividends / (1 + assets_historical_data["Stock Splits"])
    return assets_adjusted_close

def check_data(data):
    # Check if there are any NaN values or infinite values
    print("Checking data...")
    print("NaN values:", data.isna().sum())
    print("Infinite values:", (data == np.inf).sum())
    print("Min values:", data.min())
    print("Max values:", data.max())




if __name__ == "__main__":
    main()

