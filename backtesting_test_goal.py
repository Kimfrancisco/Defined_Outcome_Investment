# -*- coding: utf-8 -*-
"""
Defined Outcome ETF Backtesting Model
This code is written by Daham Kim

Backtesting Model use historical data(S&P500 and short term treasury yield)

This simulator scrape financial market data from Yahoo Finance
Web Scrapping use yahoofinancials package.
"from yahoofinancials import YahooFinancials" means importing scrapping package.
pip install yahoofinancials

yahoo finance web scrapping package return dict type return.
to see detail of return, please use pprint function.
pprint package allows you to see dictionary type data.

Option price Calculation is done by QuantLib which is opensource quant package.
pip install QuantLib

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
class data_loading:
    
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
    
    
    # scrape dividend yield of S&P500 dividend yield and convert it to annual yield 
    def dividend_yield(self):
        # dividend yield is available in quarterly at yahoo finance
        # This function allows you to download dividend yield of S&P500 stock 
        
        if ((str(self.YYYY_MM))[5:7] ) in ['01' , '02' , '03']:
            end_date = (str(self.YYYY_MM))[:4] +'-12-01'
        elif ((str(self.YYYY_MM))[5:7] ) in ['04' , '05' , '06']:
            end_date = self.next_year + '-03'
        elif ((str(self.YYYY_MM))[5:7] ) in ['07' , '08' , '09']:
            end_date = self.next_year + '-06'            
        elif ((str(self.YYYY_MM))[5:7] ) in ['10' , '11' , '12']:
            end_date = self.next_year + '-09'            
        
        
        yahoo_financials = YahooFinancials(self.ticker)
        prices = yahoo_financials.get_historical_price_data(self.begin,  self.end, 'monthly')
        price_temp =prices['SPY']['eventsData']['dividends']
        
        date_key = ([val for key, val in price_temp.items() if end_date in key])
        dividend_yield =  (float((str(date_key[0])[11:14])))
        # price_data = (prices['SPY']['eventsData']['dividends'][end_date]['amount'])/100
        
        if len(date_key) > 1:
            print("double dividend yield. look at dividend_yield function at data_loading class")
            sys.exit(1)

        return round(dividend_yield/100,4)
      
        
    # scrape short treasyry yield and convert it to annual risk free yield
    def riskfree_yield(self ):
        
        rf_ticker = '^FVX'
        yahoo_financials = YahooFinancials(rf_ticker)
        prices = yahoo_financials.get_historical_price_data(self.begin,  self.end, 'monthly')   
        
        price_data = pd.DataFrame(prices[rf_ticker]['prices'])
        price_data.index = price_data['formatted_date']
        price_data.drop(['date', 'formatted_date'], inplace = True, axis = 1)        
        
        if str(self.YYYY_MM)[:4] != '2021':
            prices_data = price_data['adjclose']
            risk_free = (prices_data.iloc[-1])/100
            
        else :
            prices_data = price_data['adjclose']
            risk_free = (prices_data.iloc[-1])/100 
        
        print("riskfree  at" , prices_data.index[-1] , "is" , round(risk_free,4))
        return risk_free
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
        

"""
option pricer use open source package so called QuantLib
This engine is built in C++ language and you can call it in python by python QuantLib
To install this package, "pip install QuantLib-Python".
To use this, you just type "import QuantLib as ql"
Now ql.~~ meanas QuantLib package function. 
Option_pricer class contains pricing algorithm for Euroepan , American Style exercise option,
using Black Scholes Merton model and Heston model.
(Heston model parameter tuning is not available for moment since, we don't have historical volatity surface data')
"""
class option_pricer:
    # declare variables requierd for pricing option
    def __init__(self, exercise_type ,option_type, position, spot , strike , calc_date , mat_date, interest, dividend, vol):
        
        self.exercise_type = exercise_type
                
        if option_type == 'Call' or option_type == 'call':
            self.option_type = ql.Option.Call
        elif option_type == 'Put' or option_type ==  'put':
            self.option_type = ql.Option.Put
        else:
            print("enter call or put as  option_type")
            sys.exit(1)
            
        self.position = position
        
        if position == 'Long' or position == 'long':
            self.position = 'Long'
        elif position == 'Short' or position == 'short':
            self.position = 'Short'
        else :
            print("enter long or short as position")
            sys.exit(1)
        

        self.strike = strike
        self.spot = spot
        
        self.calc_date = ql.Date(int(calc_date[5:7]), int(calc_date[8:10]), int(calc_date[0:4]))
        self.mat_date = ql.Date(int(mat_date[5:7]), int(mat_date[8:10]), int(mat_date[0:4]))
        
        self.interest = interest
        self.dividend = dividend
        self.vol = vol
        
        length = int(2 * self.spot)
        self.simulation_space = np.linspace(0, length, num=  length * 100 ) #0.01 단위의 시뮬레이션 array
        self.pay_off_space = np.zeros(np.size(self.simulation_space))
        
        
    # calculating Black Scholes Merton model price of option
    def bsm_price(self):
    
        transaction = 0.005 # transaction cost = 1% 임의로 하드 코딩

        # counting days to maturity with korean calendar from QuantLib Engine
        calendar = ql.UnitedStates()
        day_count = ql.Actual365Fixed()
        ql.Settings.instance().evaluationDate= self.calc_date
        
        
        if self.exercise_type == 'European' or self.exercise_type == 'european':
            payoff = ql.PlainVanillaPayoff(self.option_type, self.strike)
            exercise = ql.EuropeanExercise(self.mat_date)
            option =  ql.VanillaOption(payoff, exercise)
            
        elif self.exercise_type == 'American' or self.exercise_type == 'american':
            payoff = ql.PlainVanillaPayoff(self.option_type, self.strike)
            settlement_date = self.calc_date
            exercise = ql.AmericanExercise(settlement_date, self.mat_date)
            option =  ql.VanillaOption(payoff, exercise)
            
    
        
        #calcualting option price under Black Scholes Merton Model
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.spot))
        
        flat_vol_ts = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(self.calc_date, calendar, self.vol, day_count))
        
        flat_ts = ql.YieldTermStructureHandle(
                            ql.FlatForward(self.calc_date, self.interest, day_count) )
        
        dividend_yield = ql.YieldTermStructureHandle(
                            ql.FlatForward(self.calc_date, self.dividend, day_count))
        
        bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_yield, flat_ts, flat_vol_ts)
                
        
        if self.exercise_type == 'European' or self.exercise_type == 'european':
            engine =  ql.AnalyticEuropeanEngine(bsm_process)
            
        elif self.exercise_type == 'American' or self.exercise_type =='american':
            steps = 100
            engine =  ql.BinomialVanillaEngine(bsm_process, 'Trigeorgis', steps)
        else :
            print("exercise type should be European or American")
            sys.exit(1)
            
            
        option.setPricingEngine(engine)
        bs_price = option.NPV()
        
            
        #print("spot :" ,self.spot , " strike: ", self.strike)
        #print("The BlackScholesMerton Model price is : " , bs_price, " and positoin is ", self.position ,"\n")
    
        # if option is to be exercised, exercise price and payoff should consider transaction cost
        if self.option_type == ql.Option.Call:
            
            for i in range(np.size(self.simulation_space)):
                
                if self.simulation_space[i] > (self.strike * (1 + transaction)) :
                    self.pay_off_space[i] = round(self.simulation_space[i] - (self.strike*(1 + transaction)),2)
                else:
                    self.pay_off_space[i] = 0
                    
        elif self.option_type == ql.Option.Put:
            
            for i in range(np.size(self.simulation_space)):
                
                if self.simulation_space[i] < (self.strike*(1+transaction)):
                    self.pay_off_space[i] = round((self.strike * (1 + transaction)) - self.simulation_space[i],2)
                else:
                    self.pay_off_space[i] = 0
        
        # long option means we pay premium(bs_price)
        # short option means we receive premium(bs_price)
        if self.position == 'Long':
            self.pay_off_space = self.pay_off_space - bs_price
            bs_price = -bs_price
        else:
            self.pay_off_space = - self.pay_off_space + bs_price
            bs_price = + bs_price
        
        return round(bs_price,2) , self.pay_off_space
    
    # calculating Heston model price of option, parameter tuning yet developed
    def heston_price(self):
    
        transaction = 0.00 # transaction cost = 1% 임의로 하드 코딩
        
        # this parameters should be calculated with historical volatility surface data
        # you can fit these parameters if avaiable
        v0 = self.vol * self.vol
        kappa = 0.1
        theta = v0
        sigma = 0.1
        rho = -0.75        

        # counting days to maturity with korean calendar from QuantLib Engine
        calendar = ql.UnitedStates()
        day_count = ql.Actual365Fixed()
        ql.Settings.instance().evaluationDate= self.calc_date
        
        
        if self.exercise_type == 'European' or self.exercise_type == 'european':
            payoff = ql.PlainVanillaPayoff(self.option_type, self.strike)
            exercise = ql.EuropeanExercise(self.mat_date)
            option =  ql.VanillaOption(payoff, exercise)
            
        elif self.exercise_type == 'American' or self.exercise_type == 'american':
            payoff = ql.PlainVanillaPayoff(self.option_type, self.strike)
            settlement_date = self.calc_date
            exercise = ql.AmericanExercise(settlement_date, self.mat_date)
            option =  ql.VanillaOption(payoff, exercise)
            
    
        
        #calcualting option price under Black Scholes Merton Model
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.spot))
        
        flat_vol_ts = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(self.calc_date, calendar, self.vol, day_count))
        
        flat_ts = ql.YieldTermStructureHandle(
                            ql.FlatForward(self.calc_date, self.interest, day_count) )
        
        dividend_yield = ql.YieldTermStructureHandle(
                            ql.FlatForward(self.calc_date, self.dividend, day_count))
        
        heston_process =  ql.HestonProcess(flat_ts, dividend_yield, spot_handle, v0, kappa, theta, sigma, rho)  
        
        engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process), 0.01, 1000)
        
        
        # if self.exercise_type == 'European' or self.exercise_type == 'european':
        #     engine =  ql.AnalyticEuropeanEngine(heston_process)
            
        # elif self.exercise_type == 'American' or self.exercise_type =='american':
        #     steps = 100
        #     engine =  ql.BinomialVanillaEngine(heston_process, 'Trigeorgis', steps)
        # else :
        #     print("exercise type should be European or American")
        #     sys.exit(1)
            
            
        option.setPricingEngine(engine)
        h_price = option.NPV()
        
            
        #print("spot :" ,self.spot , " strike: ", self.strike)
        #print("The BlackScholesMerton Model price is : " , bs_price, " and positoin is ", self.position ,"\n")
    
        # if option is to be exercised, exercise price and payoff should consider transaction cost
        if self.option_type == ql.Option.Call:
            
            for i in range(np.size(self.simulation_space)):
                
                if self.simulation_space[i] > (self.strike * (1 + transaction)) :
                    self.pay_off_space[i] = round(self.simulation_space[i] - (self.strike*(1 + transaction)),2)
                else:
                    self.pay_off_space[i] = 0
                    
        elif self.option_type == ql.Option.Put:
            
            for i in range(np.size(self.simulation_space)):
                
                if self.simulation_space[i] < (self.strike*(1+transaction)):
                    self.pay_off_space[i] = round((self.strike * (1 + transaction)) - self.simulation_space[i],2)
                else:
                    self.pay_off_space[i] = 0
        
        # long option means we pay premium(bs_price)
        # short option means we receive premium(bs_price)
        if self.position == 'Long':
            self.pay_off_space = self.pay_off_space - h_price
            h_price = -h_price
        else:
            self.pay_off_space = - self.pay_off_space + h_price
            h_price = + h_price
        
        return round(h_price,2) , self.pay_off_space

    # payoff of option position
    def plain_payoff(self):
    # this function is developed soley for illustrative purpose in Jupyter Notebook
    # not yet be depreciated.
    #can be used to explain option pricing algorithm to beginers
    
        if self.option_type == ql.Option.Call:
            
            for i in range(np.size(self.simulation_space)):
                if self.simulation_space[i] - self.strike > 0:
                    self.pay_off_space[i] = round(self.simulation_space[i] - self.strike,2)
                else:
                    self.pay_off_space[i] = 0
                    
        elif self.option_type == ql.Option.Put:
            for i in range(np.size(self.simulation_space)):
                if self.simulation_space[i] < self.strike:
                    self.pay_off_space[i] = round(self.strike - self.simulation_space[i],2)
                else:
                    self.pay_off_space[i] = 0
        
        if self.position == 'Long':
            self.pay_off_space = self.pay_off_space
        else:
            self.pay_off_space = - self.pay_off_space
        return self.pay_off_space




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
