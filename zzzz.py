import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf
from pypfopt import expected_returns, EfficientFrontier, get_latest_prices
from pypfopt.discrete_allocation import DiscreteAllocation

def main():
    islamic_us_etfs = ['SPUS', 'HLAL', 'SPSK', 'SPRE', 'SPTE', 'SPWO', 'UMMA']
    assets = islamic_us_etfs

    start_date = "2020-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')

    assets_historical_data = yf.download(assets, start=start_date, end=end_date, actions=True)
    assets_adjusted_close = get_adjusted_close(assets, assets_historical_data, start_date, end_date)
    
    # Check for missing or zero values in the adjusted close
    print ("_________________________________________________________")
    print("NaN values per asset:", assets_adjusted_close.isna().sum())  # Count NaN values
    print("Zero values per asset:", (assets_adjusted_close == 0).sum())  # Count zero values

    # Replace NaN and Zero values with small numbers
    assets_adjusted_close = assets_adjusted_close.fillna(1e-8)
    assets_adjusted_close = assets_adjusted_close.replace(0, 1e-8)

    # Calculate returns using log difference
    df_returns = np.log(assets_adjusted_close).diff().dropna()

    # Market index for CAPM calculation (e.g., S&P 500)
    market_ticker = '^GSPC'  # S&P 500 index as a proxy for the market
    market_data = yf.download(market_ticker, start=start_date, end=end_date)
    market_returns = np.log(market_data['Adj Close']).diff().dropna()

    # Calculate CAPM returns for each ETF using market data
    risk_free_rate = float(get_t_bill_yield().iloc[-1]["Close"]) / 100
    retornos1 = expected_returns.capm_return(assets_adjusted_close, market_prices=market_returns, 
                                             returns_data=True, risk_free_rate=risk_free_rate, frequency=252)

    print("CAPM Expected Returns:", retornos1)

    # Calculate covariance matrix and portfolio weights
    df_aac_cov = assets_adjusted_close.cov() * 252
    pesos = pesosPortafolio(assets_adjusted_close)

    # Portfolio variance and volatility
    varianza_portafolio = pesos.T @ df_aac_cov @ pesos
    print("The variance of the portfolio is:", round(varianza_portafolio * 100, 1), "%")
    volatilidad_portafolio = np.sqrt(varianza_portafolio)
    print("The volatility of the portfolio is:", round(volatilidad_portafolio * 100, 1), "%")

    # Expected portfolio return
    retorno_portafolio = np.sum(pesos * retornos1)
    print('The expected annual return of the portfolio is:', round(retorno_portafolio * 100, 3), '%')

    # Efficient Frontier optimization
    ef = EfficientFrontier(retornos1, df_aac_cov, weight_bounds=(0, 1))
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    print(cleaned_weights)
    ef.portfolio_performance(verbose=True)

    # Discrete Allocation
    latest_prices = get_latest_prices(assets_adjusted_close)
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=10_000)
    allocation, leftover = da.lp_portfolio()
    print("Quantities of Stock To buy:", allocation)
    print("Money leftover: ${:.2f}".format(leftover))

def get_adjusted_close(assets, assets_historical_data, start_date, end_date):
    # Calculate adjusted close by subtracting dividends and considering stock splits
    assets_adjusted_close = pd.DataFrame()
    assets_close_minus_dividends = assets_historical_data["Close"] - assets_historical_data["Dividends"]
    assets_adjusted_close = assets_close_minus_dividends / (1 + assets_historical_data["Stock Splits"])
    return assets_adjusted_close

def pesosPortafolio(dataframe):
    # Initialize weights (even distribution across assets)
    return np.ones(len(dataframe.columns)) / len(dataframe.columns)

def get_t_bill_yield():
    # Fetch the 1-year Treasury rate (proxy for risk-free rate)
    ticker = '^IRX'  # 13-week Treasury bill, replace with 1-year Treasury bill if available
    t_bill_data = yf.Ticker(ticker)
    historical_data = t_bill_data.history(period='1y')
    return historical_data[['Close']].tail()

if __name__ == "__main__":
    main()
