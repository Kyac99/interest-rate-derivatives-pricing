import numpy as np
import matplotlib.pyplot as plt

# Import des modules nécessaires du projet
from models.interest_rate import Vasicek
from models.derivatives import IRSwapPricer
from instruments.swaps import InterestRateSwap

def main():
    # Paramètres du modèle de Vasicek
    r0 = 0.03       # Taux initial
    kappa = 0.5     # Vitesse de retour à la moyenne
    theta = 0.05    # Niveau moyen à long terme
    sigma = 0.01    # Volatilité
    
    # Création du modèle de taux
    vasicek_model = Vasicek(r0, kappa, theta, sigma)
    
    # Création du pricer de swap
    swap_pricer = IRSwapPricer(vasicek_model)
    
    # Paramètres du swap
    start_date = 0.0     # Aujourd'hui
    maturity_date = 5.0  # Swap de 5 ans
    payment_frequency = 0.5  # Paiements semestriels
    
    # Calcul du taux de swap à la parité
    par_rate = swap_pricer.par_rate(start_date, maturity_date, payment_frequency)
    print(f"Taux de swap à la parité: {par_rate:.4%}")
    
    # Création d'un swap avec un taux fixe différent du taux à la parité
    fixed_rate = 0.04  # 4%
    notional = 1000000  # 1 million
    
    # Création de l'objet swap
    swap = InterestRateSwap(
        start_date=start_date,
        maturity_date=maturity_date,
        fixed_rate=fixed_rate,
        payment_frequency=payment_frequency,
        notional=notional,
        is_payer=True  # Payeur de taux fixe
    )
    
    # Évaluation du swap
    swap_npv = swap.price(swap_pricer)
    print(f"NPV du swap: {swap_npv:,.2f}")
    
    # Analyse de sensibilité au taux d'intérêt
    rates = np.linspace(0.01, 0.10, 50)  # Taux de 1% à 10%
    npvs = []
    
    for rate in rates:
        # Mise à jour du taux dans le modèle
        vasicek_model.r0 = rate
        
        # Calcul du NPV avec le nouveau taux
        npv = swap.price(swap_pricer)
        npvs.append(npv)
    
    # Affichage du graphique de sensibilité
    plt.figure(figsize=(10, 6))
    plt.plot(rates * 100, npvs)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Sensibilité du NPV du swap au taux d\'intérêt')
    plt.xlabel('Taux d\'intérêt (%)')
    plt.ylabel('NPV du swap')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('swap_sensitivity.png')
    plt.close()
    
    # Restauration du taux initial
    vasicek_model.r0 = r0
    
    # Simulation de trajectoires de taux
    n_paths = 10
    rates_paths = vasicek_model.simulate_rates(n_paths=n_paths, seed=42)
    
    # Calcul de l'évolution du NPV du swap sur chaque trajectoire
    timesteps = vasicek_model.timesteps
    time_horizon = vasicek_model.time_horizon
    times = np.linspace(0, time_horizon, timesteps + 1)
    
    npv_paths = np.zeros((n_paths, timesteps + 1))
    
    for path in range(n_paths):
        for t in range(timesteps + 1):
            # Mise à jour du taux dans le modèle
            vasicek_model.r0 = rates_paths[path, t]
            
            # Ajustement du temps actuel pour l'évaluation
            current_time = times[t]
            
            # Calcul du NPV au temps t
            if current_time < maturity_date:
                npv = swap.price(swap_pricer, current_time=current_time)
                npv_paths[path, t] = npv
            else:
                npv_paths[path, t] = 0  # Swap expiré
    
    # Affichage de l'évolution du NPV sur différentes trajectoires
    plt.figure(figsize=(12, 6))
    for path in range(n_paths):
        plt.plot(times, npv_paths[path], alpha=0.7, linewidth=1)
    
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Évolution du NPV du swap sur différentes trajectoires de taux')
    plt.xlabel('Temps (années)')
    plt.ylabel('NPV du swap')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('swap_npv_paths.png')
    plt.close()
    
    print("Analyse du swap terminée. Les graphiques ont été sauvegardés.")
    
if __name__ == "__main__":
    main()