from calibration import get_calibrated_parameters
from gs_model_pricer import price_option_on_future_gibson_schwartz
import pandas as pd
import numpy as np

FILE_NAME = "Data_GS.xlsx"

try:
    # 1. On charge et on nettoie DIRECTEMENT pour être tranquille
    df = pd.read_excel(FILE_NAME)
    # On supprime les lignes où il manque soit le prix, soit le TAUX
    df = df.dropna(subset=['Close LCOc1', 'Close LCOc2', 'r_1m']).copy()
    
    # 2. On lance la calibration (en passant le chemin du fichier)
    p = get_calibrated_parameters(FILE_NAME)
    
    # 3. On prend la dernière ligne complète
    latest = df.iloc[-1]
    
    S0 = float(latest['Close LCOc1'])
    F2 = float(latest['Close LCOc2'])
    r_actuel = float(latest['r_1m']) # On s'assure qu'il est en décimal (0.03)
    
    print(f"DEBUG - Date lue : {latest['Exchange Date']}")
    print(f"DEBUG - Taux r utilisé : {r_actuel:.4f}")

    # 4. Calcul du delta0 (Yield)
    tau1, tau2 = 1/12, 2/12
    # On utilise le r_actuel de la ligne, pas celui de l'objet p qui est peut-être NaN
    delta0 = r_actuel - np.log(F2 / S0) / (tau2 - tau1)

    # 5. Pricing
    K = 80.0
    h = 0.25
    u = 0.5
    
    # On s'assure que l'objet p a bien le bon taux r pour le pricer
    # (On crée une copie avec le bon taux pour éviter le NaN)
    from dataclasses import replace
    p_final = replace(p, r=r_actuel)

    call_price = price_option_on_future_gibson_schwartz(S0, delta0, K, h, u, p_final, option="call")

    print("\n🚀 PRICING RÉEL TERMINÉ")
    print("="*50)
    print(f"ÉTAT DU MARCHÉ : Spot={S0:.2f}$ | Yield={delta0:.4f}")
    print(f"PARAMÈTRES CALIBRÉS : Kappa={p.kappa:.2f} | Rho={p.rho:.4f}")
    print("-" * 50)
    print(f"PRIX DE L'OPTION (Call {K}$ @ 3 mois) : {call_price:.4f} $")
    print("="*50)

except Exception as e:
    print(f"Erreur : {e}")

from monte_carlo_pricer import monte_carlo_gs_price

# ... (après ton calcul de call_price analytique) ...

print("\n--- CALCUL MONTE CARLO ---")
mc_price, mc_std = monte_carlo_gs_price(S0, delta0, K, h, u, p_final, n_sims=50000)

print(f"PRIX MONTE CARLO : {mc_price:.4f} $ (+/- {mc_std:.4f})")
diff = abs(call_price - mc_price)
print(f"DIFFÉRENCE ANALYTIQUE/MC : {diff:.6f} $")

if diff < 0.05:
    print("✅ VALIDATION RÉUSSIE : Les deux modèles convergent !")
else:
    print("⚠️ ATTENTION : Écart significatif détecté.")
    from fourier_pricer import price_option_fourier

# ... (après tes calculs Analytique et Monte Carlo) ...

print("\n--- CALCUL FOURIER (Inversion) ---")
try:
    # On utilise les mêmes paramètres p_final que tout à l'heure
    # On peut augmenter n_points pour plus de précision
    fourier_price = price_option_fourier(S0, delta0, K, h, u, p_final, option="call")

    print(f"PRIX FOURIER     : {fourier_price:.4f} $")
    
    diff_fourier = abs(call_price - fourier_price)
    print(f"DIFFÉRENCE ANALYTIQUE/FOURIER : {diff_fourier:.6f} $")

    if diff_fourier < 0.001:
        print("✅ VALIDATION RÉUSSIE : Fourier est quasi-identique à l'analytique !")
    else:
        print("⚠️ Écart détecté : vérifie la borne xi_max de l'intégrale.")
        
except Exception as e:
    print(f"Erreur Fourier : {e}")

print("="*50)