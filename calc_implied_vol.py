import os
import datetime as dt
import numpy as np
import scipy.optimize as optimize
import scipy.stats as stats


class implied_volatility:
    def __init__(self):
        self.volatility = -1
        self.price_difference = -1
        self.solved = False

class implied_volatility_calc:

    def __init__(self, option_type, option_price, spot_price, strike_price ,interest_rate, dividend_yield, time_to_maturity):
        self.option_type = option_type
        self.option_price = option_price
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.interest_rate = interest_rate
        self.dividend_yield = dividend_yield
        self.time_to_maturity = time_to_maturity

    def _d1(self, S0, K, r, d, sigma, T):
        return (np.log(S0 / K) + (r - d + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))

    def _d2(self, S0, K, r, d, sigma, T):
        return (np.log(S0 / K) + (r - d - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))

    def _blackscholes(self, option_type, S0, K, r, d, sigma, T):
        S0 = float(S0)
        K = float(K)
        r = float(r)
        d = float(d)
        sigma = float(sigma)
        T /= 365.0

        if option_type == "call":
            return S0 * np.exp(-d * T) * stats.norm.cdf(self._d1(S0, K, r, d, sigma, T)) \
                   - K * np.exp(-r * T) * stats.norm.cdf(self._d2(S0, K, r, d, sigma, T))
        elif option_type == "put":
            return - S0 * np.exp(-d * T) * stats.norm.cdf(-self._d1(S0, K, r, d, sigma, T)) \
                   + K * np.exp(-r * T) * stats.norm.cdf(-self._d2(S0, K, r, d, sigma, T))
        else:
            print("option_type input should be call or put")
        return 0

    def _bs_func(self, sigma):
        S0 = self.spot_price
        K = self.strike_price
        r = self.interest_rate
        d = self.dividend_yield
        T = self.time_to_maturity
        option_type = self.option_type
        return (self._blackscholes(option_type, S0, K, r, d, sigma, T) - self.option_price)

    def calc_implied_volatility(self):
        solution = implied_volatility()
        for i in range(5):
            s, infodict, ier, msg = optimize.fsolve(self._bs_func, 10, full_output=True)
            if ier == 1:
                solution.volatility = s[0]
                solution.price_difference = infodict['fvec']
                solution.solved = True
                return solution
        return solution

if __name__ == "__main__":
    aa = implied_volatility_calc(option_type = 'call', option_price = 21, spot_price= 421, strike_price=430, interest_rate=0.016, dividend_yield = 0.02, time_to_maturity=1 )
    # print("_bs_func ", aa._bs_func(sigma = 0.10))
    print(aa.calc_implied_volatility())
    result = aa.calc_implied_volatility()
    print(result)