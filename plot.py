import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace
from calibration import get_calibrated_parameters
from gs_model_pricer import futures_price, price_option_on_future_gibson_schwartz
from monte_carlo_pricer import monte_carlo_gs_price

FILE_NAME = "Data_GS.xlsx"

def generate_report_graphs():
    df = pd.read_excel(FILE_NAME)
    df = df.dropna(subset=['Close LCOc1', 'Close LCOc2', 'r_1m']).copy()
    p = get_calibrated_parameters(FILE_NAME)
    
    latest = df.iloc[-1]
    S0 = float(latest['Close LCOc1'])
    F2 = float(latest['Close LCOc2'])
    r_actuel = float(latest['r_1m'])
    p_final = replace(p, r=r_actuel)
    
    tau1, tau2 = 1/12, 2/12
    delta0 = r_actuel - np.log(F2 / S0) / (tau2 - tau1)

    print("Generating Graph 1...")
    
    market_maturities = [1/12, 2/12, 5.0]
    market_prices = [latest['Close LCOc1'], latest['Close LCOc2'], latest['Close LCOh9']]
    
   
    theoretical_maturities = np.linspace(0.01, 5.5, 100)
    theoretical_prices = [futures_price(S0, delta0, t, p_final) for t in theoretical_maturities]

    plt.figure(figsize=(10, 5))
    plt.plot(theoretical_maturities, theoretical_prices, label='Gibson-Schwartz Forward Curve', color='blue', linewidth=2)
    plt.scatter(market_maturities, market_prices, color='red', label='Market Prices (Refinitiv)', zorder=5)
    plt.title(f"Graph 1: Forward Curve Calibration ({latest['Exchange Date']})")
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('graph1_forward_curve.png')
    plt.show()

    print("Generating Graph 2...")
    df['delta_hist'] = df['r_1m'] - np.log(df['Close LCOc2'] / df['Close LCOc1']) / (tau2 - tau1)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Brent Spot Price ($)', color='tab:blue')
    ax1.plot(df['Exchange Date'], df['Close LCOc1'], color='tab:blue', label='Spot Price (LCOc1)')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Convenience Yield (delta)', color='tab:orange')
    ax2.plot(df['Exchange Date'], df['delta_hist'], color='tab:orange', alpha=0.7, label='Convenience Yield')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title("Graph 2: Historical Spot Price vs. Convenience Yield (2023-2026)")
    fig.tight_layout()
    plt.savefig('graph2_history.png')
    plt.show()

    print("Generating Graph 3 (This may take a minute)...")
    K, h, u = 80.0, 0.25, 0.5
    analytical_price = price_option_on_future_gibson_schwartz(S0, delta0, K, h, u, p_final)
    
    sim_sizes = [1000, 5000, 10000, 25000, 50000, 75000, 100000]
    mc_prices = []
    
    for size in sim_sizes:
        price, _ = monte_carlo_gs_price(S0, delta0, K, h, u, p_final, n_sims=size)
        mc_prices.append(price)

    plt.figure(figsize=(10, 5))
    plt.plot(sim_sizes, mc_prices, marker='o', linestyle='-', label='Monte Carlo Price')
    plt.axhline(y=analytical_price, color='r', linestyle='--', label=f'Analytical Price ({analytical_price:.4f}$)')
    plt.title("Graph 3: Monte Carlo Convergence Analysis")
    plt.xlabel("Number of Simulations")
    plt.ylabel("Option Price ($)")
    plt.legend()
    plt.grid(True)
    plt.savefig('graph3_convergence.png')
    plt.show()

if __name__ == "__main__":
    generate_report_graphs()