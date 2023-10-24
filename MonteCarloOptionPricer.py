import numpy as np
from enum import IntEnum

class OptionType(IntEnum):
    EUROPEAN_VANILLA = 0

class MonteCarloPricer(object):

    def __init__(self, type=OptionType.EUROPEAN_VANILLA):
        self.type=type

    def call_price_vanilla(self, num_sims, S, K, r, v, T):
        '''
        num_sims: Number of simulations to run to price the option
        S = current price of the underlying asset
        K = strike price of the option
        r = risk free interest rate
        v = volatility of the underlying asset
        T = time to expire
        '''

        S_adjusted = S * np.exp(T * (r - 0.5 * v * v))
        payoff = 0

        for _ in range(num_sims):
            gauss_rv = np.random.randn()
            s_current = S_adjusted * np.exp(np.sqrt(v * v * T) * gauss_rv)
            payoff += max(K - s_current, 0)

        return (payoff / num_sims) * np.exp(-r * T)
    
    def put_price_vanilla(self, num_sims, S, K, r, v, T):
        '''
        num_sims: Number of simulations to run to price the option
        S = current price of the underlying asset
        K = strike price of the option
        r = risk free interest rate
        v = volatility of the underlying asset
        T = time to expire
        '''

        S_adjusted = S * np.exp(T * (r - 0.5 * v * v))
        payoff = 0

        for _ in range(num_sims):
            gauss_rv = np.random.randn()
            s_current = S_adjusted * np.exp(np.sqrt(v * v * T) * gauss_rv)
            payoff += max(s_current - K, 0)

        return (payoff / num_sims) * np.exp(-r * T)
    
    def calculate_price(self, num_sims, S, K, r, v, T, option='call'):
        match option:
            case 'call':
                return self.call_price_vanilla(num_sims, S, K, r, v, T)
            case 'put':
                return self.put_price_vanilla(num_sims, S, K, r, v, T)