import numpy as np
from math import exp, sqrt
from params import GibsonSchwartzParams
from gs_model_pricer import futures_price  # On réutilise la formule du futur pour le payoff

def monte_carlo_gs_price(S0, delta0, K, h, u, p: GibsonSchwartzParams, n_sims=200000, n_steps=500):
    """
    S0, delta0 : État initial
    h : Temps jusqu'à l'exercice de l'option (ex: 0.25)
    u : Temps restant sur le futur à l'exercice (ex: 0.75)
    p : Paramètres GibsonSchwartzParams
    """
    dt = h / n_steps
    r = p.r
    k = p.kappa
    alpha = p.delta_bar_q
    s1 = p.sigma_s
    s2 = p.sigma_delta
    rho = p.rho

    # Initialisation des trajectoires
    S = np.full(n_sims, float(S0))
    delta = np.full(n_sims, float(delta0))

    # Matrice de corrélation de Cholesky
    # [dW1, dW2] avec corrélation rho
    chol = np.array([[1.0, 0.0], 
                     [rho, sqrt(1 - rho**2)]])

    for t in range(n_steps):
        # Génération de nombres aléatoires corrélés
        Z = np.random.normal(0, 1, (n_sims, 2)) @ chol.T
        dW1 = Z[:, 0] * sqrt(dt)
        dW2 = Z[:, 1] * sqrt(dt)

        # 1. Évolution du Convenience Yield (Processus OU)
        # d_delta = k * (alpha - delta) * dt + s2 * dW2
        delta += k * (alpha - delta) * dt + s2 * dW2

        # 2. Évolution du Prix Spot (Log-normal)
        # dS = (r - delta) * S * dt + s1 * S * dW1
        # On utilise la version log pour plus de stabilité numérique :
        # dlnS = (r - delta - 0.5*s1^2) * dt + s1 * dW1
        S *= np.exp((r - delta - 0.5 * s1**2) * dt + s1 * dW1)

    # --- À l'échéance h ---
    # Pour chaque simulation, on calcule le prix du futur F(h, h+u)
    # On utilise la formule analytique car l'option porte sur un FUTUR
    F_at_h = np.array([futures_price(S[i], delta[i], u, p) for i in range(n_sims)])

    # Payoff du Call : max(F - K, 0)
    payoffs = np.maximum(F_at_h - K, 0)

    # Moyenne des payoffs actualisée au taux r
    price = exp(-r * h) * np.mean(payoffs)
    
    # Erreur standard (pour ton mémoire, c'est bien de l'afficher)
    std_err = exp(-r * h) * np.std(payoffs) / sqrt(n_sims)
    
    return price, std_err