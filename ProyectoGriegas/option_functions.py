import numpy as np
from scipy.optimize import minimize
import scipy.stats as st


class BlackScholes:
    def __init__(self, St, K, T, t, r, sigma):
        """
        Initialize Black-Scholes model parameters.

        Parameters:
            St (float): Spot price
            K (float): Strike price
            T (float): Time to maturity (years)
            t (float): Current time
            r (float): Risk-free interest rate
            sigma (float): Volatility of the underlying asset
        """
        self.St = St
        self.K = K
        self.T = T
        self.t = t
        self.r = r
        self.sigma = sigma

    # Black-Scholes formula components
    def d1(self):
        """Compute d1 in the Black-Scholes formula."""
        return (np.log(self.St / self.K) + (self.r + 0.5 * self.sigma ** 2) * (self.T - self.t)) / (self.sigma * np.sqrt(self.T - self.t))

    def d2(self):
        """Compute d2 in the Black-Scholes formula."""
        return (np.log(self.St / self.K) + (self.r - 0.5 * self.sigma ** 2) * (self.T - self.t)) / (self.sigma * np.sqrt(self.T - self.t))

    # Cumulative distribution function for a standard normal variable
    @staticmethod
    def N(x):
        """Cumulative distribution function for a standard normal variable."""
        return st.norm.cdf(x)
    
    # Option pricing methods
    def binary_cash_or_nothing_put(self):
        """Price of a Binary Cash-or-Nothing Put option."""
        return self.K * np.exp(-self.r * (self.T - self.t)) * self.N(-self.d2())

    def binary_cash_or_nothing_call(self):
        """Price of a Binary Cash-or-Nothing Call option."""
        return self.K * np.exp(-self.r * (self.T - self.t)) * self.N(self.d2())
    
    def binary_asset_or_nothing_put(self):
        """Price of a Binary Asset-or-Nothing Put option."""
        return self.St * self.N(-self.d1())

    def binary_asset_or_nothing_call(self):
        """Price of a Binary Asset-or-Nothing Call option."""
        return self.St * self.N(self.d1())
    
    # Financial options pricing methods (European options)
    def financial_call(self):
        """Price of a European Call option."""
        return self.binary_asset_or_nothing_call() - self.binary_cash_or_nothing_call()
    
    def financial_put(self):
        """Price of a European Put option."""
        return self.binary_cash_or_nothing_put() - self.binary_asset_or_nothing_put()
    
    # Non binary options pricing methods
    def ECall(self):
        """Price of a European Call option."""
        return self.St * self.N(self.d1()) - self.K * np.exp(-self.r * (self.T - self.t)) * self.N(self.d2())
    
    def EPut(self):
        """Price of a European Put option."""
        return self.K * np.exp(-self.r * (self.T - self.t)) * self.N(-self.d2()) - self.St * self.N(-self.d1())
    
    # Greeks options pricing methods
    def delta_call(self):
        """Delta of a European Call option."""
        return self.N(self.d1())
    
    def delta_put(self):
        """Delta of a European Put option."""
        return self.N(self.d1()) - 1
    
    def gamma(self):
        """Gamma of the option."""
        return st.norm.pdf(self.d1()) / (self.St * self.sigma * np.sqrt(self.T - self.t))
    
    def vega(self):
        """Vega of the option."""
        return self.St * st.norm.pdf(self.d1()) * np.sqrt(self.T - self.t)
    
    def theta_call(self):
        """Theta of a European Call option."""
        return (-self.St * st.norm.pdf(self.d1()) * self.sigma / (2 * np.sqrt(self.T - self.t)) - self.r * self.K * np.exp(-self.r * (self.T - self.t)) * self.N(self.d2()))
    
    def theta_put(self):
        """Theta of a European Put option."""
        return (-self.St * st.norm.pdf(self.d1()) * self.sigma / (2 * np.sqrt(self.T - self.t)) + self.r * self.K * np.exp(-self.r * (self.T - self.t)) * self.N(-self.d2()))
    
    def rho_call(self):
        """Rho of a European Call option."""
        return self.K * (self.T - self.t) * np.exp(-self.r * (self.T - self.t)) * self.N(self.d2())
    
    def rho_put(self):
        """Rho of a European Put option."""
        return -self.K * (self.T - self.t) * np.exp(-self.r * (self.T - self.t)) * self.N(-self.d2())

    @staticmethod
    def calculate_greeks_df(df, option_type='call'):
        """
        Calculate the 5 Greeks for each row in a DataFrame.
        
        Parameters:
            df (pd.DataFrame): DataFrame with columns 'St', 'K', 'T', 't', 'r', 'sigma'
            option_type (str): Either 'call' or 'put'
        
        Returns:
            pd.DataFrame: Original DataFrame with added columns for Delta, Gamma, Vega, Theta, and Rho
        """
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("option_type must be either 'call' or 'put'")
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Initialize lists to store Greeks
        deltas = []
        gammas = []
        vegas = []
        thetas = []
        rhos = []
        
        # Calculate Greeks for each row
        for idx, row in df.iterrows():
            bs = BlackScholes(
                St=row['St'],
                K=row['K'],
                T=row['T'],
                t=row['t'],
                r=row['r'],
                sigma=row['sigma']
            )
            
            if option_type.lower() == 'call':
                deltas.append(bs.delta_call())
                thetas.append(bs.theta_call())
                rhos.append(bs.rho_call())
            else:  # put
                deltas.append(bs.delta_put())
                thetas.append(bs.theta_put())
                rhos.append(bs.rho_put())
            
            # Gamma and Vega are the same for both calls and puts
            gammas.append(bs.gamma())
            vegas.append(bs.vega())
        
        # Add Greeks as new columns
        result_df['Delta'] = deltas
        result_df['Gamma'] = gammas
        result_df['Vega'] = vegas
        result_df['Theta'] = thetas
        result_df['Rho'] = rhos
        
        return result_df
    

def imp_vol_call(St, K, T, t, r, sigma, CallMkt):
    """
    Placeholder for implied volatility calculation function.
    """
    bounds = [[None, None]]
    apply_constraints1 = lambda x: BlackScholes(St, K, T, t, r, x[0]).ECall() - CallMkt

    my_constraints = {'type': 'eq', 'fun': apply_constraints1}

    a = minimize(lambda x: 0.0,
                x0=np.array([sigma]),
                bounds=bounds,
                constraints=my_constraints,
                method='SLSQP')

    ImpSigma = a.x[0]

    return ImpSigma


def imp_vol_put(St, K, T, t, r, sigma, CallMkt):
    """
    Placeholder for implied volatility calculation function.
    """
    bounds = [[None, None]]
    apply_constraints1 = lambda x: BlackScholes(St, K, T, t, r, x[0]).EPut() - CallMkt

    my_constraints = {'type': 'eq', 'fun': apply_constraints1}

    a = minimize(lambda x: 0.0,
                x0=np.array([sigma]),
                bounds=bounds,
                constraints=my_constraints,
                method='SLSQP')

    ImpSigma = a.x[0]

    return ImpSigma
