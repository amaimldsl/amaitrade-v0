import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf
from pypfopt import expected_returns, EfficientFrontier, risk_models

def main():
    islamic_us_etfs = ['SPUS', 'HLAL', 'SPSK', 'SPRE', 'SPTE', 'SPWO', 'UMMA']
    assets = islamic_us_etfs

    start_date = "2020-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')

    assets_historical_data = yf.download(assets, start=start_date, end=end_date, actions=True)
    assets_adjusted_close = get_adjusted_close(assets, assets_historical_data, start_date, end_date)

    # Calculate expected returns and sample covariance
    mu = expected_returns.capm_return(assets_adjusted_close)
    S = risk_models.sample_cov(assets_adjusted_close)

    # Optimize for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    print(cleaned_weights)

    # Calculate the expected performance
    ef.portfolio_performance(verbose=True)

    # Calculate the portfolio return manually
    retorno_portafolio = np.dot(list(cleaned_weights.values()), mu)
    print(f"Manual Portfolio Return: {retorno_portafolio}")

def get_adjusted_close(assets, assets_historical_data, start_date, end_date):
    assets_adjusted_close = pd.DataFrame()
    assets_close_minus_dividends = assets_historical_data["Close"] - assets_historical_data["Dividends"]
    assets_adjusted_close = assets_close_minus_dividends / (1 + assets_historical_data["Stock Splits"])
    return assets_adjusted_close

if __name__ == "__main__":
    main()