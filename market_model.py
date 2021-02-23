# -*- coding: utf-8 -*-
"""
Defined Outcome ETF Backtesting Model
This code is written by Daham Kim

Complete simulation consists of 1) market modeling 2) fund modeling 3) guarantee calculation

This module is market modeling part (first part).
Whloe function and classes declared and used can be called main package.
To use this , from market_model import class_name.function_name.
Market Modeling Module use historical data(S&P500)

This simulator scrape financial market data from Yahoo Finance
Web Scrapping use yahoofinancials package.
!pip install yahoofinancials

yahoo finance web scrapping package return dict type return.
to see detail of return, please use pprint function.
pprint package allows you to see dictionary type data.

historic data use Geometric Brownian Motion to capture statistical characteristics
of financial market(S&P500) 

Market Data is calcualted in annual form and converted to monthly form afterward.
These data is calculated using log form and thus log-normal distribution is assumed.

"""


"""
Other required packages  pandas, numpy, matplotlib.pyplot , random, math, os, sys
"""
import pandas as pd
import numpy as np
import random
import math
import os
import sys
import matplotlib.pyplot as plt

os.chdir("C:\\Users\kimfr\Google Drive\Milliman\ETF_2\Data")

import QuantLib as ql

from yahoofinancials import YahooFinancials
from pprint import pprint as pp


"""
financial data loading part
data_laoding use web scrapping method from "yahoo finance"
this class contains data downloading of
daily_price_data, monthly_price_data of S&P500 index
riskfree rate(US short term treasury yield)
dividend yiled of S&P500 composites
VIX index as standard deviation of S&P500 index
"""
class historical_data_loading:
    
    def __init__(self, YYYY_MM ):
        
        self.YYYY_MM = YYYY_MM
        self.ticker = 'SPY'
        
        #begin은 1일을 기준으로 다운로드
        #end period는 월별 말일에 맞추어서 들어감
        self.begin = YYYY_MM +'-01'
        
        self.next_year =  str(int(str(YYYY_MM)[0:4]) + 1)
        self.end = (self.next_year) + str(YYYY_MM)[4:7] +'-01'
        
    
    # scrape daily price data 
    def daily_price_data_loading(self):
        yahoo_financials = YahooFinancials(self.ticker)
        prices = yahoo_financials.get_historical_price_data(self.begin, self.end, 'daily')
        
        price_data = pd.DataFrame(prices[self.ticker]['prices'])
        price_data.index = price_data['formatted_date']
        price_data.drop(['date', 'formatted_date'], inplace = True, axis = 1)
        
        return price_data
    
    # scrape monthly price data
    def monthly_price_data_loading(self):
        yahoo_financials = YahooFinancials(self.ticker)
        prices = yahoo_financials.get_historical_price_data(self.begin, self.end, 'monthly')
        
        price_data = pd.DataFrame(prices[self.ticker]['prices'])
        price_data.index = price_data['formatted_date']
        price_data.drop(['date', 'formatted_date'], inplace = True, axis = 1)
        return price_data
    
    
    # vix is used to measure volatility of market. VIX so called fear index is 30 day forward implied volatility 
    def vix(self):
        # VIX so called fear index is 30 day forward implied volatility 
        # we use VIX as implied volatility (annualized)
        # VIX / 100  = annualuzed 30-days implied volatility
        
        VIX_ticker = '^VIX'
        yahoo_financials = YahooFinancials(VIX_ticker)
        prices = yahoo_financials.get_historical_price_data(self.begin,  self.end, 'monthly')   
        
        price_data = pd.DataFrame(prices[VIX_ticker]['prices'])
        price_data.index = price_data['formatted_date']
        price_data.drop(['date', 'formatted_date'], inplace = True, axis = 1)        
        
        if str(self.YYYY_MM)[:4] != '2021':
            prices_data = price_data['adjclose']
            vix_rate = (prices_data.iloc[-1])/100
            
        else :
            prices_data = price_data['adjclose']
            vix_rate = (prices_data.iloc[-1])/100 
        
        vix_rate = round(vix_rate,4)
        print("vix  at" , prices_data.index[-1] , "is" , round(vix_rate,4))
        
        return vix_rate
    
    
    def print_self(self):
        print(self.begin, self.end)
        print("For test purpose")
        
class GBM:
    
    def __init__(self, start_YYYY_MM , end_YYYY_MM):
        
        self.start = start_YYYY_MM
        self.end   = end_YYYY_MM
        self.ticker = 'SPY'
        
        #begin은 1일을 기준으로 다운로드
        #end period는 월별 말일에 맞추어서 들어감
        # self.begin = YYYY_MM +'-01'
        
        # self.next_year =  str(int(str(YYYY_MM)[0:4]) + 1)
        # self.end = (self.next_year) + str(YYYY_MM)[4:7] +'-01'
        
        if   int(str(end_YYYY_MM)[0:4]) == int(str(start_YYYY_MM)[0:4]) and int(str(end_YYYY_MM)[5:7]) > int(str(start_YYYY_MM)[5:7]):
            self.period = int(str(end_YYYY_MM)[5:7]) - int(str(start_YYYY_MM)[5:7])
        elif int(str(end_YYYY_MM)[0:4]) == int(str(start_YYYY_MM)[0:4]) and int(str(end_YYYY_MM)[5:7]) < int(str(start_YYYY_MM)[5:7]):
            print("Start YYYY_MM cannot be earlier than End YYYY_MM")
            sys.exit(1)
        elif int(str(end_YYYY_MM)[0:4]) == int(str(start_YYYY_MM)[0:4]) and int(str(end_YYYY_MM)[5:7]) == int(str(start_YYYY_MM)[5:7]):
             print("Start YYYY_MM and End YYYY_MM is same, please check input date")
            sys.exit(1)
        else:
            self.period = int(str(end_YYYY_MM)[0:4]) - int(str(start_YYYY_MM)[0:4])
        
    def random_generation()

# importing statistics of S&P500 index be DataLoading class
# this function nowhere belongs to class.
# global method. just call it.
def stats(name_space):
    
    ex = name_space.daily_price_data_loading()
    spot = round(ex['adjclose'][-1] , 2)
    
    
    #logret means return of index we want to follow
    logret = np.log(ex['adjclose']/ex['adjclose'].shift(1))
    
    # average and std of S&P index
    mu =  round(logret.mean() * ex.shape[0] , 4) # expected return
    sigma =name_space.vix() # measuring implied volatility with vix index

    # required information, risk_free rate and dividend yield
    risk_free = round(name_space.riskfree_yield(),4)
    dividend  =  round(name_space.dividend_yield(),4)
    
    print("spot" , spot)
    print("mu"   , mu)
    print("std"  , sigma)
    print("rf"   , risk_free)
    print("div"  , dividend)
    
    return spot , mu, sigma, risk_free , dividend, ex['adjclose']

# 3X up, 1x down, enhanced growth index
# _simle method is finding volatility surface using historical cap data
def enhance_buffer_etf_smile(date):
    """
    I cannot find this etf's historical data from web
    to measure volatility surface of this options, you need historical data to calcuate cap
    if you find historical data, insert it to 'CAP_Monthly.xlsx', sheet_name = 'enhance_history_cap'
    otherwise, this function will not work
    """
    history = pd.read_excel('CAP_Monthly.xlsx', sheet_name = 'enhance_history_cap',index_col = 'Date')
    cap = 1 + history.loc[date]['Cap']
    print("implied volatility : " , sigma)
    print("cap rate :" , cap)
    
    for x in range(1000000):
        
        volatility_smile = [1 + random.random(), 1 + random.random(), 1 + random.random() ,1 + random.random(), 1 + random.random() , 1 + random.random() ]

        if volatility_smile[0] > volatility_smile[1] + 0.50:
            condition1 = True
        else: condition1 = False
        if volatility_smile[0] > volatility_smile[3]:
            condition2 = True
        else: condition2 = False
        if volatility_smile[3] > volatility_smile[2] + 0.20:
            condition3 = True
        else: condition3 = False
        if volatility_smile[2] > volatility_smile[5]:
            condition4 = True
        else: condition4 = False
        if volatility_smile[4] < volatility_smile[5] + 0.20:
            condition5 = True
        else: condition6 = False
        
        if condition1 == False or condition2 == False or condition3 == False or condition4 == False or condition5 == False : continue    
        
        a_layer = option_pricer('American','Call','Long' , spot, 0.60 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, (volatility_smile[0] + 0.3) * sigma)
        b_layer = option_pricer('American','Put' ,'Short', spot, 0.60 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, (volatility_smile[0] + 0.1) * sigma)
        c_layer = option_pricer('American','Put' ,'Long' , spot, 1.20 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, (volatility_smile[0] + 0.0) * sigma)
        d_layer = option_pricer('American','Call','Short', spot, 1.20 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, (volatility_smile[0] + 0.2) * sigma)
        e_layer = option_pricer('American','Call','Long' , spot, 1.00 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, (volatility_smile[0] + 0.0) * sigma)
        
        # GoalSeek 구조를 통해서 H_layer의 TBD strike를 정해야
        h_layer = option_pricer('American','Call','Short', spot, cap  * spot , '2021-01-01', '2022-01-01', risk_free, dividend, (volatility_smile[0] + 0.1) * sigma)
        
        a , aa = a_layer.bsm_price()
        b , bb = b_layer.bsm_price()
        c , cc = c_layer.bsm_price()
        d , dd = d_layer.bsm_price()
        e , ee = e_layer.bsm_price()

        h , hh = h_layer.bsm_price()
        
        total_option_position_cost = ((2 * a) + (2 * b) + c + d + (2*e) +  (3 * h)) 
        total_option_position_payoff = (2 * aa) + (2 * bb) + cc + dd + (2*ee) + (3 * hh)
       
        # print(int(spot) , "/" , spot ,"payoff at spot", total_option_position_payoff[spot*100] )
        print(x , "/ 1000000 :" ,"innitial scale :", round(100 * (-total_option_position_cost /spot) , 2 ) ,"% ")
       
        if abs(total_option_position_cost + spot) / spot < 0.001 :
            print("spot   :" , spot )
            print("strike :" , cap * spot )
            print("smile  :" , volatility_smile)
            print('\n')
            
            print(a,b,c,d,e,h)
            print((2*a) + (2*b) + c + d + e + h + spot)
            
            return volatility_smile
            break
    return 0

def enhance_buffer_etf():
    
    print("implied volatility : " , sigma)
    print("volatility smile   : " , smile)
    
    for x in range(1000):
        multiple = (x+1000)/1000
        
        a_layer = option_pricer('American','Call','Long' , spot, 0.6 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        b_layer = option_pricer('American','Put' ,'Short', spot, 0.6 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        c_layer = option_pricer('American','Put' ,'Long' , spot, 1.2 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        d_layer = option_pricer('American','Call','Short', spot, 1.2 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        e_layer = option_pricer('American','Call','Long' , spot, 1.00* spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        
        # GoalSeek 구조를 통해서 H_layer의 TBD strike를 정해야
        h_layer = option_pricer('American','Call','Short', spot, multiple * spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        
        a , aa = a_layer.bsm_price()
        b , bb = b_layer.bsm_price()
        c , cc = c_layer.bsm_price()
        d , dd = d_layer.bsm_price()
        e , ee = e_layer.bsm_price()

        h , hh = h_layer.bsm_price()
        
        total_option_position_cost = ((2 * a) + (2 * b) + c + d + (2*e) +  (3 * h)) 
        total_option_position_payoff = (2 * aa) + (2 * bb) + cc + dd + (2*ee) + (3 * hh)
       
        # print(int(spot) , "/" , spot ,"payoff at spot", total_option_position_payoff[spot*100] )
        print(x , "/ 1000 : multiple =" ,multiple ,"innitial scale :", round(100 * (-total_option_position_cost /spot),2) ,"% ")
       
        if abs(total_option_position_cost + spot) / spot < 0.001 :
            print("goal seek result " , multiple)
            print("spot   :" ,spot)
            print('\n')

            print(a,b,c,d,e,h)
            print((2*a) + (2*b) + c + d + (2*e) + (3*h) + spot)
            
            return volatility_smile
            break
    return 0

# _simle method is finding volatility surface using historical cap data
def buffer_etf_smile(date):
    history = pd.read_excel('CAP_Monthly.xlsx', sheet_name = 'power_history_cap',index_col = 'Date')
    cap = 1 + history.loc[date]['Cap']
    print("implied volatility : " , sigma)
    print("cap rate :" , cap)
    
    for x in range(1000000): 
        
        volatility_smile = [1 + random.random(), 1 + random.random(), 1 + random.random() ,1 + random.random(), 1 + random.random() , 1 + random.random() ,1 + random.random()]

        if volatility_smile[0] > volatility_smile[1] + 0.5 :
            condition1 = True
        else: condition1 = False
        if volatility_smile[0] > volatility_smile[3]:
            condition2 = True
        else: condition2 = False
        if volatility_smile[3] > volatility_smile[2] + 0.3:
            condition3 = True
        else: condition3 = False
        if volatility_smile[2] < volatility_smile[5]:
            condition4 = True
        else: condition4 = False
        if volatility_smile[2] < volatility_smile[6]:
            condition5 = True
        else: condition5 = False        
        if volatility_smile[3] > volatility_smile[6]:
            condition6 = True
        else: condition6 = False
        if volatility_smile[5] - volatility_smile[6] < 0.05:
            condition7 = True
        else: condition7 = False
        
        if condition1 == False or condition2 == False or condition3 == False or condition4 == False or condition5 == False or condition7 == False : continue    

        a_layer = option_pricer('American','Call','Long' , spot, 0.6 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        b_layer = option_pricer('American','Put' ,'Short', spot, 0.6 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        c_layer = option_pricer('American','Put' ,'Long' , spot, 1.2 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        d_layer = option_pricer('American','Call','Short', spot, 1.2 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        e_layer = option_pricer('American','Call','Long' , spot, 1.00* spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        f_layer = option_pricer('American','Put' ,'Short', spot, 0.65* spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        g_layer = option_pricer('American','Put' ,'Long' , spot, 0.95* spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        
        # GoalSeek 구조를 통해서 H_layer의 TBD strike를 정해야
        h_layer = option_pricer('American','Call','Short', spot, cap * spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        
        a , aa = a_layer.bsm_price()
        b , bb = b_layer.bsm_price()
        c , cc = c_layer.bsm_price()
        d , dd = d_layer.bsm_price()
        e , ee = e_layer.bsm_price()
        f , ff = f_layer.bsm_price()
        g , gg = g_layer.bsm_price()

        h , hh = h_layer.bsm_price()
        
        total_option_position_cost = ((2 * a) + (2 * b) + c + d + e + f + g + (2 * h)) 
        total_option_position_payoff = (2 * aa) + (2 * bb) + cc + dd + ee + ff + gg + (2 * hh)         

        print(x , "/ 1000000 :" ,"innitial scale :", round(100 * (-total_option_position_cost /spot) , 2 ) ,"% ")
       
        if abs(total_option_position_cost + spot) / spot < 0.001 :
            print("spot   :" , spot )
            print("strike :" , cap * spot )
            print("smile  :" , volatility_smile)
            print('\n')
            
            print(a,b,c,d,e,h)
            print((2*a) + (2*b) + c + d + e + h + spot)
            
            return volatility_smile
            break
    return 0

# _etf method is to calcuate payoff structure using given volatily smile
def buffer_etf(smile):

    print("implied volatility : " , sigma)
    print("volatility smile   : " , smile)
    
    for x in range(1000):
        multiple = (x+1000)/1000
        a_layer = option_pricer('American','Call','Long' , spot, 0.6 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, smile[0] * sigma)
        b_layer = option_pricer('American','Put' ,'Short', spot, 0.6 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, smile[1] * sigma)
        c_layer = option_pricer('American','Put' ,'Long' , spot, 1.2 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, smile[2] * sigma)
        d_layer = option_pricer('American','Call','Short', spot, 1.2 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, smile[3] * sigma)
        e_layer = option_pricer('American','Call','Long' , spot, 1.00* spot , '2021-01-01', '2022-01-01', risk_free, dividend, smile[4] * sigma)
        f_layer = option_pricer('American','Put' ,'Short', spot, 0.65* spot , '2021-01-01', '2022-01-01', risk_free, dividend, smile[5] * sigma)
        g_layer = option_pricer('American','Put' ,'Long' , spot, 0.95*spot  , '2021-01-01', '2022-01-01', risk_free, dividend, smile[6] * sigma)
        
        # GoalSeek 구조를 통해서 H_layer의 TBD strike를 정해야
        h_layer = option_pricer('American','Call','Short', spot, multiple * spot , '2021-01-01', '2022-01-01', risk_free, dividend, smile[7] *sigma)
        
        a , aa = a_layer.bsm_price()
        b , bb = b_layer.bsm_price()
        c , cc = c_layer.bsm_price()
        d , dd = d_layer.bsm_price()
        e , ee = e_layer.bsm_price()
        f , ff = f_layer.bsm_price()
        g , gg = g_layer.bsm_price()

        h , hh = h_layer.bsm_price()
        
        total_option_position_cost = ((2 * a) + (2 * b) + c + d + e + f + g + (2 * h)) 
        total_option_position_payoff = (2 * aa) + (2 * bb) + cc + dd + ee + ff + gg + (2 * hh)
        
        # print(int(spot) , "/" , spot ,"payoff at spot", total_option_position_payoff[spot*100] )
        print(x , "/ 1000 : multiple =" ,multiple ,"innitial scale :", round(100 * (-total_option_position_cost /spot),2) ,"% ")
       
        if abs(total_option_position_cost + spot) / spot < 0.001 :
            print('\n')
            print("goal seek result " , multiple )
            print("spot   :" ,spot)
            print("strike :" ,multiple * spot)
            print(a,b,c,d,f,g,h)
            break
    return total_option_position_payoff
    
# _simle method is finding volatility surface using historical cap data
def power_buffer_etf_smile(date):
    history = pd.read_excel('CAP_Monthly.xlsx', sheet_name = 'power_history_cap',index_col = 'Date')
    cap = 1 + history.loc[date]['Cap']
    print("implied volatility : " , sigma)
    print("cap rate :" , cap)
    
    for x in range(1000000):
        
        volatility_smile = [1 + random.random(), 1 + random.random(), 1 + random.random() ,1 + random.random(), 1 + random.random() , 1 + random.random() , 1 + random.random() ]

        if volatility_smile[0] > volatility_smile[1] + 0.5 :
            condition1 = True
        else: condition1 = False
        if volatility_smile[0] > volatility_smile[3]:
            condition2 = True
        else: condition2 = False
        if volatility_smile[3] > volatility_smile[2] + 0.3:
            condition3 = True
        else: condition3 = False
        if volatility_smile[2] > volatility_smile[5]:
            condition4 = True
        else: condition4 = False
        if volatility_smile[3] > volatility_smile[6]:
            condition5 = True
        else: condition5 = False
        if volatility_smile[4] - volatility_smile[5] < 0.10:
            condition6 = True
        else: condition6 = False
        
        if condition1 == False or condition2 == False or condition3 == False or condition4 == False or condition5 == False or condition6 == False: continue
        
        a_layer = option_pricer('American','Call','Long' , spot, 0.6 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, (volatility_smile[0] + 0.3) * sigma)
        b_layer = option_pricer('American','Put' ,'Short', spot, 0.6 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, (volatility_smile[1] + 0.1) * sigma)
        c_layer = option_pricer('American','Put' ,'Long' , spot, 1.2 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, (volatility_smile[2] + 0.0) * sigma)
        d_layer = option_pricer('American','Call','Short', spot, 1.2 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, (volatility_smile[3] + 0.2) * sigma)
        f_layer = option_pricer('American','Put' ,'Short', spot, 0.85* spot , '2021-01-01', '2022-01-01', risk_free, dividend, (volatility_smile[4] + 0.0) * sigma)
        g_layer = option_pricer('American','Put' ,'Long' , spot, 1.0 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, (volatility_smile[5] + 0.0) * sigma)
        
        # GoalSeek 구조를 통해서 H_layer의 TBD strike를 정해야
        h_layer = option_pricer('American','Call','Short', spot, cap * spot , '2021-01-01', '2022-01-01', risk_free, dividend, (volatility_smile[6] + 0.1) * sigma)
        
        a , aa = a_layer.bsm_price()
        b , bb = b_layer.bsm_price()
        c , cc = c_layer.bsm_price()
        d , dd = d_layer.bsm_price()
        f , ff = f_layer.bsm_price()
        g , gg = g_layer.bsm_price()
        
        h , hh = h_layer.bsm_price()
        
        total_option_position_cost = (2 * a) + (2 * b) + c + d + f + g + h 
        total_option_position_payoff = (2 * aa) + (2 * bb) + cc + dd + ff + gg + hh
        
        # print(int(spot) , "/" , spot ,"payoff at spot", total_option_position_payoff[spot*100] )
        
        print(x , "/ 1000000 :" ,"innitial scale :", round(100 * (-total_option_position_cost /spot) , 2 ) ,"% ")
        
        if abs(total_option_position_cost + spot) / spot < 0.001 :
            print("spot   :" , spot )
            print("strike :" , cap * spot )
            print("smile  :" , volatility_smile, '\n')
            
            print(a,b,c,d,f,g,h)
            print((2*a) + (2*b) + c + d + f + g + h + spot)
            
            return volatility_smile
            break
    return 0
      
def power_buffer_etf(smile):
    
    print("implied volatility : " , sigma)
    print("volatility smile   : " , smile)
    
    for x in range(1000):
        multiple = (x+1000)/1000
        a_layer = option_pricer('American','Call','Long' , spot, 0.6 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, smile[0] * sigma)
        b_layer = option_pricer('American','Put' ,'Short', spot, 0.6 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, smile[1] * sigma)
        c_layer = option_pricer('American','Put' ,'Long' , spot, 1.2 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, smile[2] * sigma)
        d_layer = option_pricer('American','Call','Short', spot, 1.2 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, smile[3] * sigma)
        f_layer = option_pricer('American','Put' ,'Short', spot, 0.85* spot , '2021-01-01', '2022-01-01', risk_free, dividend, smile[4] * sigma)
        g_layer = option_pricer('American','Put' ,'Long' , spot, 1.0 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, smile[5] * sigma)
        
        # GoalSeek 구조를 통해서 H_layer의 TBD strike를 정해야
        h_layer = option_pricer('American','Call','Short', spot, multiple * spot , '2021-01-01', '2022-01-01', risk_free, dividend, smile[6] * sigma)
        
        a , aa = a_layer.bsm_price()
        b , bb = b_layer.bsm_price()
        c , cc = c_layer.bsm_price()
        d , dd = d_layer.bsm_price()
        f , ff = f_layer.bsm_price()
        g , gg = g_layer.bsm_price()
        
        h , hh = h_layer.bsm_price()
        
        total_option_position_cost = (2 * a) + (2 * b) + c + d + f + g + h 
        total_option_position_payoff = (2 * aa) + (2 * bb) + cc + dd + ff + gg + hh
        
        # print(int(spot) , "/" , spot ,"payoff at spot", total_option_position_payoff[spot*100] )
        print(x , "/ 1000 : multiple =" ,multiple ,"innitial scale :", round(100 * (-total_option_position_cost /spot),2) ,"% ")
       
        if abs(total_option_position_cost + spot) / spot < 0.001 :
            print('\n')
            print("goal seek result " , multiple )
            print("spot   :" ,spot)
            print("strike :" ,multiple * spot , '\n')
            
            print(a,b,c,d,f,g,h)
            print((2*a) + (2*b) + c + d + f + g + h + spot)
            
            break
        
    return total_option_position_payoff

def power_buffer_etf_heston(smile):
    
    print("implied volatility : " , sigma)
    
    for x in range(100):
        multiple = (x+100)/100
        a_layer = option_pricer('American','Call','Long' , spot, 0.60 * spot     , '2021-01-01', '2022-01-01', risk_free, dividend,      smile[0] * sigma)
        b_layer = option_pricer('American','Put' ,'Short', spot, 0.60 * spot     , '2021-01-01', '2022-01-01', risk_free, dividend,      smile[1] * sigma)
        c_layer = option_pricer('American','Put' ,'Long' , spot, 1.20 * spot     , '2021-01-01', '2022-01-01', risk_free, dividend,      smile[2] * sigma)
        d_layer = option_pricer('American','Call','Short', spot, 1.20 * spot     , '2021-01-01', '2022-01-01', risk_free, dividend,      smile[3] * sigma)
        f_layer = option_pricer('American','Put' ,'Short', spot, 0.85 * spot     , '2021-01-01', '2022-01-01', risk_free, dividend,      smile[4] * sigma)
        g_layer = option_pricer('American','Put' ,'Long' , spot, 1.00 * spot     , '2021-01-01', '2022-01-01', risk_free, dividend,      smile[5] * sigma)
        
        # GoalSeek 구조를 통해서 H_layer의 TBD strike를 정해야
        h_layer = option_pricer('American','Call','Short', spot, multiple * spot , '2021-01-01', '2022-01-01', risk_free, dividend,      smile[6] * sigma)
        
        a , aa = a_layer.heston_price()
        b , bb = b_layer.heston_price()
        c , cc = c_layer.heston_price()
        d , dd = d_layer.heston_price()
        f , ff = f_layer.heston_price()
        g , gg = g_layer.heston_price()
        
        h , hh = h_layer.heston_price()
        
        total_option_position_cost = (2 * a) + (2 * b) + c + d + f + g + h 
        total_option_position_payoff = (2 * aa) + (2 * bb) + cc + dd + ff + gg + hh
        
        # print(int(spot) , "/" , spot ,"payoff at spot", total_option_position_payoff[spot*100] )
        print(x , "/ 1000 : multiple =" ,multiple ,"innitial scale :", round(100 * (- total_option_position_cost /spot),2) ,"% ")
       
        if abs(total_option_position_cost + spot) / spot < 0.005 :
            print('\n')
            print("goal seek result " , multiple )
            print("spot   :" ,spot)
            print("strike :" ,multiple * spot)
            print("sigma :" , sigma * multiple , '\n')
            print(a,b,c,d,f,g,h)
            print((2*a) + (2*b) + c + d + f + g + h + spot)
            
            break
        
    return total_option_position_payoff

# _simle method is finding volatility surface using historical cap data
def ultra_buffer_etf_smile(date):
    
    history = pd.read_excel('CAP_Monthly.xlsx', sheet_name = 'power_history_cap',index_col = 'Date')
    cap = 1 + history.loc[date]['Cap']
    print("implied volatility : " , sigma)
    print("cap rate :" , cap)
    
    for x in range(1000000):
        
        volatility_smile = [1 + random.random(), 1 + random.random(), 1 + random.random() ,1 + random.random(), 1 + random.random() , 1 + random.random() ]

        if volatility_smile[0] > volatility_smile[1] + 0.50:
            condition1 = True
        else: condition1 = False
        if volatility_smile[0] > volatility_smile[3]:
            condition2 = True
        else: condition2 = False
        if volatility_smile[3] > volatility_smile[2] + 0.20:
            condition3 = True
        else: condition3 = False
        if volatility_smile[2] > volatility_smile[5]:
            condition4 = True
        else: condition4 = False
        if volatility_smile[4] > volatility_smile[5] + 0.05:
            condition5 = True
        else: condition6 = False
        
        if condition1 == False or condition2 == False or condition3 == False or condition4 == False or condition5 == False : continue    
    
        a_layer = option_pricer('American','Call','Long' , spot, 0.6  * spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        b_layer = option_pricer('American','Put' ,'Short', spot, 0.6  * spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        c_layer = option_pricer('American','Put' ,'Long' , spot, 1.2  * spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        d_layer = option_pricer('American','Call','Short', spot, 1.2  * spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        f_layer = option_pricer('American','Put' ,'Short', spot, 0.65 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        g_layer = option_pricer('American','Put' ,'Long' , spot, 0.95 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        
        # GoalSeek 구조를 통해서 H_layer의 TBD strike를 정해야
        h_layer = option_pricer('American','Call','Short', spot, cap  * spot , '2021-01-01', '2022-01-01', risk_free, dividend, sigma)
        
        a , aa = a_layer.bsm_price()
        b , bb = b_layer.bsm_price()
        c , cc = c_layer.bsm_price()
        d , dd = d_layer.bsm_price()
        f , ff = f_layer.bsm_price()
        g , gg = g_layer.bsm_price()
        
        h , hh = h_layer.bsm_price()
        
        total_option_position_cost = ((2 * a) + (2 * b) + c + d + f + g + h) 
        total_option_position_payoff = (2 * aa) + (2 * bb) + cc + dd + ff + gg + hh    
       
def ultra_buffer_etf(smile):   
    
    for x in range(1000):
        multiple = (x+1000)/1000
        a_layer = option_pricer('American','Call','Long' , spot, 0.6 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, smile[0] * sigma)
        b_layer = option_pricer('American','Put' ,'Short', spot, 0.6 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, smile[1] * sigma)
        c_layer = option_pricer('American','Put' ,'Long' , spot, 1.2 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, smile[2] * sigma)
        d_layer = option_pricer('American','Call','Short', spot, 1.2 * spot , '2021-01-01', '2022-01-01', risk_free, dividend, smile[3] * sigma)
        f_layer = option_pricer('American','Put' ,'Short', spot, 0.65* spot , '2021-01-01', '2022-01-01', risk_free, dividend, smile[4] * sigma)
        g_layer = option_pricer('American','Put' ,'Long' , spot, 0.95 * spot ,'2021-01-01', '2022-01-01', risk_free, dividend, smile[5] * sigma)
        
        # GoalSeek 구조를 통해서 H_layer의 TBD strike를 정해야
        h_layer = option_pricer('American','Call','Short', spot, multiple * spot , '2021-01-01', '2022-01-01', risk_free, dividend, smile[6] * sigma)
        
        a , aa = a_layer.bsm_price()
        b , bb = b_layer.bsm_price()
        c , cc = c_layer.bsm_price()
        d , dd = d_layer.bsm_price()
        f , ff = f_layer.bsm_price()
        g , gg = g_layer.bsm_price()
        
        h , hh = h_layer.bsm_price()
        
        total_option_position_cost = ((2 * a) + (2 * b) + c + d + f + g + h) 
        total_option_position_payoff = (2 * aa) + (2 * bb) + cc + dd + ff + gg + hh
        
        # print(int(spot) , "/" , spot ,"payoff at spot", total_option_position_payoff[spot*100] )
        print(x , "/ 1000 : multiple =" ,multiple ,"innitial scale :", round(100 * (-total_option_position_cost /spot),2) ,"% ")
       
        if abs(total_option_position_cost + spot) / spot < 0.001 :
            print('\n')
            print("goal seek result " , multiple )
            print("spot   :" ,spot)
            print("strike :" ,multiple * spot)
            break
    return total_option_position_payoff
 



class risk_analytic:
    def __init__(self, data , pay, spot, mu, sigma):
        # before var class, all statistical characteristic of option is calculated in annual basis
        # now convert it to monthly basis
        self.data = data
        self.mu = (mu/12) + 0.5 * (sigma * sigma / 12)
        self.sigma = sigma/(12 ** 0.5)
        
        
        # brownian_movement dataframe's row    = time to maturity(month)(default = 10 years)
        # brownian_movement dataframe's column = scenario (default = 10000)
        brownian_movement = pd.DataFrame(columns = np.linspace(1,100,100), index = np.linspace(1,120,120) )
        
        for i in range (len(brownian_movement.columns)):
            
            np.random.seed(seed= i )
            random = np.random.normal(loc = 0.0 , scale = (1/np.sqrt(12)) , size = len(brownian_movement.index))
            
            for j in range (len(brownian_movement.index)):
                
                np.random.normal(loc = self.mu , scale = self.sigma)
                brownian_movement.iloc[i,j] = math.exp( ((self.mu - 0.5 * self.sigma * self.sigma)/12 ) + (self.sigma * random[j]))
        
        self.stoch_scenario =  brownian_movement.pct_change()
        self.stoch_scenario.fillna(1, inplace = True)
        
    def day_value_at_risk_1_percent(self):
        from scipy.stats import skew, kurtosis, kurtosistest
        from scipy.stats import norm
         
        data_values = np.array(self.data.values)  # daily adj-close prices
        ret = data_values[1:]/data_values[:-1] - 1    # compute daily returns
         
        # N(x; mu, sig) best fit (finding: mu, stdev)
        mu_norm, sig_norm = norm.fit(ret)
        dx = 0.0001  # resolution
        x = np.arange(-0.1, 0.1, dx)
        
        pdf = norm.pdf(x, mu_norm, sig_norm)
        print("Sample mean  = : " , mu_norm)
        print("Sample stdev = : " , sig_norm)
        print('\n')
        
        # Compute VaRs and CVaRs
         
        h = 1  # time interval of VAR, CVAR
        alpha = 0.01  # significance level
        lev = 100*(1-alpha)
         
        CVaR_n = alpha**-1 * norm.pdf(norm.ppf(alpha))*sig_norm - mu_norm
        VaR_n = norm.ppf(1-alpha)*sig_norm - mu_norm
         
         
        print("%g%% %g-day Normal VaR     = %.2f%%" % (lev, h, VaR_n*100))
        print("%g%% %g-day Normal t CVaR  = %.2f%%" % (lev, h, CVaR_n*100))
        
        return round(VaR_n , 4) , round(CVaR_n , 4)
        
    def day_value_at_risk_5_percent(self):
        from scipy.stats import skew, kurtosis, kurtosistest
        from scipy.stats import norm
         
        data_values = np.array(self.data.values)  # daily adj-close prices
        ret = data_values[1:]/data_values[:-1] - 1    # compute daily returns
         
        # N(x; mu, sig) best fit (finding: mu, stdev)
        mu_norm, sig_norm = norm.fit(ret)
        dx = 0.0001  # resolution
        x = np.arange(-0.1, 0.1, dx)
        
        pdf = norm.pdf(x, mu_norm, sig_norm)
        print("Sample mean  = : " , mu_norm)
        print("Sample stdev = : " , sig_norm)
        print('\n')
        
        # Compute VaRs and CVaRs
         
        h = 1  # time interval of VAR, CVAR
        alpha = 0.05  # significance level
        lev = 100*(1-alpha)
         
        CVaR_n = alpha**-1 * norm.pdf(norm.ppf(alpha))*sig_norm - mu_norm
        VaR_n = norm.ppf(1-alpha)*sig_norm - mu_norm
         
         
        print("%g%% %g-day Normal VaR   = %.2f%%" % (lev, h, VaR_n*100))
        print("%g%% %g-day Normal CVaR  = %.2f%%" % (lev, h, CVaR_n*100))
        
        return round(VaR_n , 4) , round(CVaR_n , 4)

    def day_value_at_risk_10_percent(self):
        from scipy.stats import skew, kurtosis, kurtosistest
        from scipy.stats import norm
         
        data_values = np.array(self.data.values)  # daily adj-close prices
        ret = data_values[1:]/data_values[:-1] - 1    # compute daily returns
         
        # N(x; mu, sig) best fit (finding: mu, stdev)
        mu_norm, sig_norm = norm.fit(ret)
        dx = 0.0001  # resolution
        x = np.arange(-0.1, 0.1, dx)
        
        pdf = norm.pdf(x, mu_norm, sig_norm)
        print("Sample mean  = : " , mu_norm)
        print("Sample stdev = : " , sig_norm)
        print('\n')
        
        # Compute VaRs and CVaRs
         
        h = 1  # time interval of VAR, CVAR
        alpha = 0.10  # significance level
        lev = 100*(1-alpha)
         
        CVaR_n = alpha**-1 * norm.pdf(norm.ppf(alpha))*sig_norm - mu_norm
        VaR_n = norm.ppf(1-alpha)*sig_norm - mu_norm
         
         
        print("%g%% %g-day Normal VaR     = %.2f%%" % (lev, h, VaR_n*100))
        print("%g%% %g-day Normal t CVaR  = %.2f%%" % (lev, h, CVaR_n*100))
        
        return round(VaR_n , 4) , round(CVaR_n , 4)

    def day_value_at_risk_15_percent(self):
        from scipy.stats import skew, kurtosis, kurtosistest
        from scipy.stats import norm
         
        data_values = np.array(self.data.values)  # daily adj-close prices
        ret = data_values[1:]/data_values[:-1] - 1    # compute daily returns
         
        # N(x; mu, sig) best fit (finding: mu, stdev)
        mu_norm, sig_norm = norm.fit(ret)
        dx = 0.0001  # resolution
        x = np.arange(-0.1, 0.1, dx)
        
        pdf = norm.pdf(x, mu_norm, sig_norm)
        print("Sample mean  = : " , mu_norm)
        print("Sample stdev = : " , sig_norm)
        print('\n')
        
        # Compute VaRs and CVaRs
         
        h = 1  # time interval of VAR, CVAR
        alpha = 0.15  # significance level
        lev = 100*(1-alpha)
         
        CVaR_n = alpha**-1 * norm.pdf(norm.ppf(alpha))*sig_norm - mu_norm
        VaR_n = norm.ppf(1-alpha)*sig_norm - mu_norm
         
         
        print("%g%% %g-day Normal VaR     = %.2f%%" % (lev, h, VaR_n*100))
        print("%g%% %g-day Normal t CVaR  = %.2f%%" % (lev, h, CVaR_n*100))
        
        return round(VaR_n , 4) , round(CVaR_n , 4)




"""
SP01 = data_loading('2001-01')
SP02 = data_loading('2002-01')
SP03 = data_loading('2003-01')
SP04 = data_loading('2004-01')
SP05 = data_loading('2005-01')
SP06 = data_loading('2006-01')
SP07 = data_loading('2007-01')
SP08 = data_loading('2008-01')
SP09 = data_loading('2009-01')
SP10 = data_loading('2010-01')
SP11 = data_loading('2011-01')
SP12 = data_loading('2012-01')
SP13 = data_loading('2013-01')
SP14 = data_loading('2014-01')
SP15 = data_loading('2015-01')
SP16 = data_loading('2016-01')
SP17 = data_loading('2017-01')
SP18 = data_loading('2018-01')
SP19 = data_loading('2019-01')
"""

# main function
if __name__ == "__main__":  
    
    SP = data_loading(('2019-01'))
    spot, mu, sigma, risk_free, dividend , data = stats(SP)
    smile_list = power_buffer_etf_smile(1901)
    
    SP = data_loading(('2020-01'))
    spot, mu, sigma, risk_free, dividend , data = stats(SP)
    
    pay   = power_buffer_etf(smile_list)
    
    simu = np.linspace(0,int(spot*2) ,int(spot*2) * 100 )
    
    plt.plot( simu[20000:50000] , pay[20000:50000])
