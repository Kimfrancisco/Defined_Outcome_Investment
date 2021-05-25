import pandas as pd
import numpy as np
from scipy.stats import norm

class option_pricing:
    """
    calculate fair price of option with given market data using black-sholes merton model
    several market data is required

    Parameters
    ----------
    option_type : str
        'call' or 'put'
    trade_date : str
        'YYYY-MM-DD' form date on which option is traded.
        in real case, both trade date and settlement date is needed to calculate exact price.
        However, we assume trade is made and settled immediately
    maturity_date : str
        'YYYY-MM-DD' form
    time_to_maturity : float
        time left to maturity in annual form
    risk_free : float
        risk free yield is needed. for US market we use
    dividend_yield : float
        dividend yield of underlying stock is needed.
        in robust pricing, we use expected dividend yield based on dividend swap.
        However, we will use historical data instead
    spot_price : float
        spot price at trade date
    strike_price : float
        strike price of option

    implied_vol : float
        implied volatility of option
        have to consider volatility smile with different strike price

    """
    def __init__(self, option_type, position, time_to_maturity,spot_price,strike_price,risk_free,dividend_yield,implied_vol):
        self.option_type = option_type
        self.time_to_maturity = time_to_maturity
        self.position = position

        self.spot_price = spot_price
        self.spot_price_norm = 100
        self.strike_price = strike_price
        self.strike_price_norm = 100 * (strike_price / spot_price)

        self.risk_free = risk_free
        self.dividend_yield = dividend_yield

        self.implied_vol = implied_vol
        self.unit = 0.01

    def bsm(self):
        d1 = (np.log(self.spot_price/self.strike_price) + (self.risk_free - self.dividend_yield + (0.5*self.implied_vol*self.implied_vol)) * self.time_to_maturity) / (self.implied_vol * np.sqrt(self.time_to_maturity))
        d2 = d1 - (self.implied_vol * np.sqrt(self.time_to_maturity))

        if self.option_type == 'call':
            self.opt_price = self.spot_price * np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(d1) - self.strike_price * np.exp(-self.risk_free * self.time_to_maturity) * norm.cdf(d2)
        elif self.option_type == 'put':
            self.opt_price = self.strike_price * np.exp(-self.risk_free * self.time_to_maturity) * norm.cdf(-d2) - self.spot_price * np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(-d1)

        return self.opt_price

    def plain_payoff(self):
        payoff_space = np.zeros(int(200 / self.unit))
        if self.option_type == 'call':
            i = 0
            while (i * self.unit) < (self.strike_price_norm):
                payoff_space[i] = 0
                i = i + 1
            while (i * self.unit) > (self.strike_price_norm):
                if i * self.unit < 200:
                    payoff_space[i] = (i*self.unit) - self.strike_price_norm
                    i = i + 1
                else:
                    break
        elif self.option_type == 'put':
            i = 0
            while (i * self.unit) < (self.strike_price_norm):
                payoff_space[i] = self.strike_price_norm - (i*self.unit)
                i = i + 1
            while (i * self.unit) > (self.strike_price_norm):
                if i * self.unit < 200:
                    payoff_space[i] = 0
                    i = i + 1
                else:
                    break
        else:
            print("option type should be 'call' or 'put', please check option type input")

        self.payoff_space = payoff_space
        return self.payoff_space

    def option_payoff(self):
        if self.position == 'long':
            return self.plain_payoff() - (self.bsm() / self.spot_price)
        elif self.position == 'short':
            return - self.plain_payoff() + (self.bsm() /self.spot_price)
        else:
            print("option position is not valid")
        return 0

class calc_implied_volatility:
    def __init__(self , ):
        self.option_type = option_type
        self.time_to_maturity = time_to_maturity

        self.spot_price = spot_price
        self.spot_price_norm = 100
        self.strike_price = strike_price
        self.strike_price_norm = 100 * (strike_price / spot_price)

        self.risk_free = risk_free
        self.dividend_yield = dividend_yield

        self.unit = 0.01

# main function
if __name__ == "__main__":

    aa = pricing.option_pricing(option_type='call', position='long', time_to_maturity=1, spot_price=421,
                                strike_price=400, risk_free=0.0165, dividend_yield=0.02, implied_vol=0.10)

    print(aa.plain_payoff())
    line = (aa.plain_payoff())




