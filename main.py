import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import statistics
import yfinance as yf
import seaborn as sns
from pandas_datareader import data
from pulp import *
from  pypfopt import expected_returns, EfficientFrontier, get_latest_prices
from pypfopt.discrete_allocation import DiscreteAllocation


def main():
    islamic_us_etfs = ['SPUS', 'HLAL', 'SPSK', 'SPRE', 'SPTE', 'SPWO', 'UMMA']
    top_tech = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA', 'PYPL', 'ADBE', 'INTC', 'CSCO']
    top_etfs = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VEA', 'VWO', 'VTV', 'VUG', 'VOO']
    top_comodities_etfs = ['GLD', 'SLV', 'USO','UNG','PPLT','PALL','WEAT','CORN','DBA', 'DBB', 'DBC', 'DBO', 'DBP']
    #assets = islamic_us_etfs
    assets = islamic_us_etfs  + top_tech + top_etfs + top_comodities_etfs 
    start_date = "2016-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')

    assets_historical_data = yf.download(assets, start=start_date, end= end_date ,actions=True)
    assets_adjusted_close = get_adjusted_close(assets, assets_historical_data, start_date, end_date)
            
    #replace NAN with 0
    #assets_adjusted_close=assets_adjusted_close.fillna(0)

    
    #plt.figure(figsize=(12.2,4.5)) 
    #for i in assets_adjusted_close.columns.values:
    #    plt.plot( assets_adjusted_close[i],  label=i)
    #plt.title('Price of the Stocks')
    #plt.xlabel('Date',fontsize=18)
    #plt.ylabel('Price in USD',fontsize=18)
    ##plt.legend(assets_adjusted_close.columns.values, loc='upper left')
    #plt.show()


    #---
    #df_draw = np.log(assets_adjusted_close).diff()
    #df_draw = df_draw.dropna()

    #plt.figure(figsize=(12.2,4.5)) 
    #for i in df_draw.columns.values:
    #    plt.hist( df_draw[i],  label=i, bins = 200)
    #plt.title('Returns Histogram')
    #plt.xlabel('Fecha',fontsize=18)
    #plt.ylabel('Precio en USD',fontsize=18)
    #plt.legend(df_draw.columns.values)
    #plt.show()


    #df_assets =  df.loc[:, df.columns != '^IXIC']
    #df_benchmark1 =  df.loc[:, df.columns == '^IXIC']

    
    
    risk_free_rate = float(get_t_bill_yield().iloc[-1]["Close"])/100

    #risk_free_rate = .07/100

    assets_adjusted_close = assets_adjusted_close.dropna()
    assets_adjusted_close_log = np.log(assets_adjusted_close).diff()
    
  
    retornos1 = expected_returns.capm_return(assets_adjusted_close_log , market_prices = None , returns_data= True, risk_free_rate=risk_free_rate, frequency=252)
    
    df_aac_cov = assets_adjusted_close_log.cov()*252
    
    pesos = pesosPortafolio(assets_adjusted_close_log)

    


    print (retornos1)

    print ("____________________________With Similar Shares____________________________________")
    #Portfolio Variance:
    varianza_portafolio = pesos.T @ df_aac_cov @pesos 
    print ("The variance of the portfolio is:" + " " + str(round(varianza_portafolio*100,1))+"%")


    # Portfolio Volatility
    volatilidad_portafolio = np.sqrt(varianza_portafolio)
    print ("The volatility of the portfolio is:" + " " + str(round(volatilidad_portafolio*100,1))+"%")


    # Expected return of the portfolio
    retorno_portafolio = np.sum(pesos*retornos1)
    print('The expected annual return of the portfolio is:' + ' ' + str(round(retorno_portafolio*100,3)) + '%')

    print ("____________________________With EF Shares____________________________________")
    ef = EfficientFrontier(retornos1, df_aac_cov,weight_bounds=(0,1),verbose=True)
    weights = ef.max_sharpe() 
    cleaned_weights = ef.clean_weights() 
    print(cleaned_weights) 
    ef.portfolio_performance(verbose=True)

    
    latest_prices = get_latest_prices(assets_adjusted_close)
    pesos = cleaned_weights 
    da = DiscreteAllocation(pesos, latest_prices, total_portfolio_value=100)
    allocation, leftover = da.lp_portfolio()
    print("Quantities of Stock To buy:", allocation)
    print("Money leftover: ${:.2f}".format(leftover))
        
def get_adjusted_close(assets, assets_historical_data,start_date, end_date):
    assets_adjsted_close = pd.DataFrame()
    assets_close_minus_dividends = assets_historical_data["Close"] - assets_historical_data["Dividends"]
    assets_adjsted_close = assets_close_minus_dividends / (1 + assets_historical_data["Stock Splits"])
    return assets_adjsted_close


def pesosPortafolio(dataframe):
    array = []
    for i in dataframe.columns:
        array.append(1/len(dataframe.columns))
    arrayFinal = np.array(array)
    return arrayFinal

def get_t_bill_yield():
    # Use the 1-year Treasury rate symbol (proxy)
    ticker = '^IRX'  # Ticker for 13-week Treasury bill, use another symbol if available for 1-year T-bill

    # Fetch the data
    t_bill_data = yf.Ticker(ticker)

    # Get historical data (e.g., last 1 year)
    historical_data = t_bill_data.history(period='1y')

    # Display the most recent T-Bill yield
    return historical_data[['Close']].tail()


if __name__ == "__main__":
    main()