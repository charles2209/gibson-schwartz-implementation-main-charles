import pandas as pd
import numpy as np
import statsmodels.api as sm
from params import GibsonSchwartzParams

def get_calibrated_parameters(file_path):
    # 1. Load clean Excel data
    df = pd.read_excel(file_path)
    
    # Ensure data is sorted by date
    df = df.sort_values('Exchange Date').reset_index(drop=True)

    # 2. Calculate instantaneous Convenience Yield (delta)
    # Formula: delta = r - ln(F2/F1) / (tau2 - tau1)
    tau1, tau2 = 1/12, 2/12  # 1-month and 2-month maturities
    df['delta'] = df['r_1m'] - np.log(df['Close LCOc2'] / df['Close LCOc1']) / (tau2 - tau1)

    # 3. Calculate long-term target (delta_bar_q)
    # Using the distant contract LCOh9 (approx. 5-year proxy)
    tau_h9 = 5.0 # Adjust according to the actual maturity of your H9 contract
    df['delta_star_series'] = df['r_5y'] - np.log(df['Close LCOh9'] / df['Close LCOc1']) / (tau_h9 - tau1)
    delta_bar_q = df['delta_star_series'].mean()

    # 4. Regression for Kappa (Mean Reversion Speed)
    # Y = daily variation, X = previous day's level (lagged)
    df['delta_diff'] = df['delta'].diff()
    df['delta_lag'] = df['delta'].shift(1)
    
    reg_data = df[['delta_diff', 'delta_lag']].dropna()
    
    Y = reg_data['delta_diff']
    X = sm.add_constant(reg_data['delta_lag'])
    
    model = sm.OLS(Y, X).fit()
    a, b = model.params['const'], model.params['delta_lag']
    
    kappa = -b * 252 # Annualization
    # alpha_implied = a / -b  # Optional: alpha implied by the regression

    # 5. Calculate Volatilities (sigma)
    # Sigma_delta: standard deviation of regression residuals
    sigma_delta = model.resid.std() * np.sqrt(252)
    
    # Sigma_S: spot price volatility (using LCOc1 as proxy)
    df['spot_ret'] = np.log(df['Close LCOc1'] / df['Close LCOc1'].shift(1))
    sigma_s = df['spot_ret'].std() * np.sqrt(252)

    # 6. Correlation (rho)
    rho = df['spot_ret'].corr(df['delta_diff'])

    # Return object with all calibrated parameters
    return GibsonSchwartzParams(
        r=df['r_1m'].iloc[-1], # Use the last known rate for initial pricing setup
        kappa=kappa,
        delta_bar_q=delta_bar_q,
        sigma_s=sigma_s,
        sigma_delta=sigma_delta,
        rho=rho
    )