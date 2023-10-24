
import numpy as np
import scipy.stats
from enum import IntEnum

class OptionType(IntEnum):
    EUROPEAN_VANILLA = 0
    EUROPEAN_FUTURES = 1
    EUROPEAN_STOCK_INDEX = 2
    EUROPEAN_CURRENCY = 3

class BlackScholesPricer(object):
    def __init__(self, type=OptionType.EUROPEAN_FUTURES):
        self.type = type
    
    def generalised_black_scholes(self, S, T, K, r, b, sigma, option='call'):

        # b for European options = r 
        # b for European stock options with continuous dividend = r - q where q is the dividend payout
        # b for European futures options = 0
        # b for European currency options = r - rj where rj is the risk free rate of the foreign currency

        # Calculating d1

        d1 = (np.log(S/K) + T * (b + (sigma ** 2) / 2)) / (sigma * np.sqrt(T))

        # Calculating d2

        d2 = d1 - sigma * np.sqrt(T)

        # Differentiating between put and call

        if option == 'call':
            return S * np.exp(T * (b - r)) * scipy.stats.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * scipy.stats.norm.cdf(d2, 0.0, 1.0)
        elif option == 'put':
            return K * np.exp(-r * T) * scipy.stats.norm.cdf(-d2, 0.0, 1.0) - S * np.exp(T * (b - r)) * scipy.stats.norm.cdf(-d1, 0.0, 1.0)

    def calculate_price(self, S, T, K, r, sigma, dividend = 0, risk_free_rate_of_currency=0, option='call'):
        match self.type:
            case OptionType.EUROPEAN_VANILLA:
                price = self.generalised_black_scholes(S, T, K, r, r, sigma, option)
            case OptionType.EUROPEAN_FUTURES:
                price = self.generalised_black_scholes(S, T, K, r, 0, sigma, option)
            case OptionType.EUROPEAN_CURRENCY:
                price = self.generalised_black_scholes(S, T, K, r, r-risk_free_rate_of_currency, sigma, option)
            case OptionType.EUROPEAN_STOCK_INDEX:
                price = self.generalised_black_scholes(S, T, K, r, r-dividend, sigma, option)

        return price