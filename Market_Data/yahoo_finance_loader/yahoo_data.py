import pandas as pd
from yahoofinancials import YahooFinancials

def monthly_price_data_loading(begin, end , ticker):
    yahoo_financials = YahooFinancials(ticker)
    print("----- Downloading Monthly Data {} from Yahoo Finance -----".format(ticker))
    prices = yahoo_financials.get_historical_price_data(begin, end, 'monthly')
    
    price_data = pd.DataFrame(prices[ticker]['prices'])
    price_data.index = price_data['formatted_date']
    price_data.drop(['date', 'formatted_date'], inplace = True, axis = 1)
    
    return price_data

def daily_price_data_loading(begin, end , ticker):
    yahoo_financials = YahooFinancials(ticker)
    print("----- Downloading Daily Data {} from Yahoo Finance -----".format(ticker))
    prices = yahoo_financials.get_historical_price_data(begin, end, 'daily')
    
    price_data = pd.DataFrame(prices[ticker]['prices'])
    price_data.index = price_data['formatted_date']
    price_data.drop(['date', 'formatted_date'], inplace = True, axis = 1)
    
    return price_data