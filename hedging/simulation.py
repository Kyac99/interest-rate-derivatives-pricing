import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class HedgingSimulator:
    """
    Simulateur pour la couverture dynamique des dérivés de taux.
    
    Cette classe permet de simuler et d'analyser différentes stratégies de
    couverture pour les produits dérivés de taux d'intérêt.
    """
    
    def __init__(self, rate_model, instrument, hedging_strategy, rebalance_frequency=0.1):
        """
        Initialise le simulateur de hedging.
        
        Args:
            rate_model: Modèle de taux d'intérêt
            instrument: Instrument dérivé à couvrir
            hedging_strategy: Stratégie de couverture à utiliser
            rebalance_frequency (float): Fréquence de rééquilibrage de la couverture en années
        """
        self.rate_model = rate_model
        self.instrument = instrument
        self.hedging_strategy = hedging_strategy
        self.rebalance_frequency = rebalance_frequency
    
    def simulate(self, n_paths=100, time_horizon=None, seed=None, show_progress=True):
        """
        Exécute la simulation de couverture sur plusieurs trajectoires.
        
        Args:
            n_paths (int): Nombre de trajectoires à simuler
            time_horizon (float): Horizon temporel en années (par défaut: selon le rate_model)
            seed (int): Graine pour la génération aléatoire
            show_progress (bool): Afficher une barre de progression
            
        Returns:
            dict: Résultats de la simulation
        """
        if time_horizon is None:
            time_horizon = self.rate_model.time_horizon
            
        # Adaptation du modèle de taux si nécessaire
        if time_horizon != self.rate_model.time_horizon:
            timesteps = int(time_horizon / self.rate_model.dt)
            self.rate_model.time_horizon = time_horizon
            self.rate_model.timesteps = timesteps
        
        # Simulation des trajectoires de taux
        rates = self.rate_model.simulate_rates(n_paths=n_paths, seed=seed)
        
        # Paramètres de temps
        timesteps = self.rate_model.timesteps
        dt = self.rate_model.dt
        times = np.linspace(0, time_horizon, timesteps + 1)
        
        # Calcul des dates de rééquilibrage
        rebalance_steps = []
        current_time = 0
        while current_time <= time_horizon:
            idx = int(current_time / dt)
            if idx <= timesteps:
                rebalance_steps.append(idx)
            current_time += self.rebalance_frequency
        
        # Initialisation des résultats
        instrument_values = np.zeros((n_paths, timesteps + 1))
        hedge_values = np.zeros((n_paths, timesteps + 1))
        hedge_ratios = np.zeros((n_paths, timesteps + 1))
        pnl = np.zeros((n_paths, timesteps + 1))
        hedge_costs = np.zeros((n_paths, timesteps + 1))
        
        # Simulation du hedging pour chaque trajectoire
        iterator = tqdm(range(n_paths)) if show_progress else range(n_paths)
        
        for path in iterator:
            # Valeurs initiales
            self.rate_model.r0 = rates[path, 0]
            
            # Valeur initiale de l'instrument
            instrument_values[path, 0] = self.instrument.price(
                self.hedging_strategy.pricer, valuation_date=times[0]
            )
            
            # Calcul du ratio de couverture initial
            hedge_ratio = self.hedging_strategy.compute_hedge_ratio(times[0], rates[path, 0])
            hedge_ratios[path, 0] = hedge_ratio
            
            # Valeur initiale de la couverture
            hedge_values[path, 0] = hedge_ratio * rates[path, 0]  # Simplification
            
            # PnL initial (zéro)
            pnl[path, 0] = 0
            
            # Simulation sur la trajectoire
            for t in range(1, timesteps + 1):
                # Mise à jour du taux
                self.rate_model.r0 = rates[path, t]
                
                # Valeur de l'instrument
                instrument_values[path, t] = self.instrument.price(
                    self.hedging_strategy.pricer, valuation_date=times[t]
                )
                
                # Valeur de la couverture avant rééquilibrage
                hedge_values[path, t] = hedge_ratios[path, t-1] * rates[path, t]  # Simplification
                
                # Calcul du PnL
                pnl_t = (instrument_values[path, t] - instrument_values[path, t-1]) - \
                        (hedge_values[path, t] - hedge_values[path, t-1])
                
                # Rééquilibrage si nécessaire
                if t in rebalance_steps:
                    old_hedge_ratio = hedge_ratios[path, t-1]
                    new_hedge_ratio = self.hedging_strategy.compute_hedge_ratio(times[t], rates[path, t])
                    
                    # Coût de transaction pour le rééquilibrage
                    transaction_cost = abs(new_hedge_ratio - old_hedge_ratio) * 0.0001 * rates[path, t]
                    hedge_costs[path, t] = transaction_cost
                    
                    # Mise à jour du ratio de couverture
                    hedge_ratios[path, t] = new_hedge_ratio
                    
                    # Impact du coût sur le PnL
                    pnl_t -= transaction_cost
                else:
                    # Pas de rééquilibrage
                    hedge_ratios[path, t] = hedge_ratios[path, t-1]
                
                # Mise à jour du PnL cumulé
                pnl[path, t] = pnl[path, t-1] + pnl_t
        
        # Résultats de la simulation
        results = {
            'times': times,
            'rates': rates,
            'instrument_values': instrument_values,
            'hedge_ratios': hedge_ratios,
            'hedge_values': hedge_values,
            'hedge_costs': hedge_costs,
            'pnl': pnl
        }
        
        return results
    
    def analyze_results(self, results, confidence_level=0.95, path_indices=None):
        """
        Analyse les résultats de la simulation de couverture.
        
        Args:
            results (dict): Résultats de la simulation
            confidence_level (float): Niveau de confiance pour le calcul des quantiles
            path_indices (list): Indices des trajectoires à afficher (None pour utiliser l'ensemble)
            
        Returns:
            dict: Statistiques et analyses
        """
        pnl = results['pnl']
        n_paths = pnl.shape[0]
        timesteps = pnl.shape[1] - 1
        
        # Statistiques finales
        final_pnl = pnl[:, -1]
        mean_pnl = np.mean(final_pnl)
        std_pnl = np.std(final_pnl)
        
        # Calcul des quantiles pour la VaR et l'Expected Shortfall
        alpha = 1 - confidence_level
        var = np.percentile(final_pnl, alpha * 100)
        es = np.mean(final_pnl[final_pnl <= var])
        
        # Ratio de Sharpe (approximatif)
        sharpe_ratio = mean_pnl / std_pnl if std_pnl > 0 else np.nan
        
        # Statistiques sur les coûts de transaction
        total_hedge_costs = np.sum(results['hedge_costs'], axis=1)
        mean_hedge_costs = np.mean(total_hedge_costs)
        
        # Résultats de l'analyse
        analysis = {
            'mean_pnl': mean_pnl,
            'std_pnl': std_pnl,
            'var': var,
            'expected_shortfall': es,
            'sharpe_ratio': sharpe_ratio,
            'mean_hedge_costs': mean_hedge_costs,
            'final_pnl_distribution': final_pnl
        }
        
        return analysis
    
    def plot_results(self, results, path_indices=None, figsize=(12, 10)):
        """
        Affiche les graphiques des résultats de la simulation.
        
        Args:
            results (dict): Résultats de la simulation
            path_indices (list): Indices des trajectoires à afficher (None pour en choisir aléatoirement)
            figsize (tuple): Taille des figures
            
        Returns:
            tuple: Figures et axes matplotlib
        """
        n_paths = results['pnl'].shape[0]
        
        # Sélection des trajectoires à afficher
        if path_indices is None:
            # Sélection aléatoire de 5 trajectoires
            path_indices = np.random.choice(n_paths, min(5, n_paths), replace=False)
        
        # Création des figures
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # 1. Trajectoires de taux
        for idx in path_indices:
            axes[0].plot(results['times'], results['rates'][idx], alpha=0.7)
        
        axes[0].set_title('Trajectoires de taux d\'intérêt')
        axes[0].set_ylabel('Taux')
        axes[0].grid(True)
        
        # 2. Valeurs de l'instrument et de la couverture
        for idx in path_indices:
            axes[1].plot(results['times'], results['instrument_values'][idx], alpha=0.7, label=f'Instrument {idx}')
            axes[1].plot(results['times'], results['hedge_values'][idx], '--', alpha=0.7, label=f'Couverture {idx}')
        
        axes[1].set_title('Valeurs de l\'instrument et de la couverture')
        axes[1].set_ylabel('Valeur')
        axes[1].grid(True)
        if len(path_indices) <= 2:  # Afficher la légende seulement si peu de trajectoires
            axes[1].legend()
        
        # 3. PnL de la stratégie de couverture
        for idx in path_indices:
            axes[2].plot(results['times'], results['pnl'][idx], alpha=0.7)
        
        # Ajouter le PnL moyen
        mean_pnl = np.mean(results['pnl'], axis=0)
        axes[2].plot(results['times'], mean_pnl, 'k--', linewidth=2, label='PnL moyen')
        
        axes[2].set_title('Profit & Loss (PnL) de la stratégie de couverture')
        axes[2].set_xlabel('Temps (années)')
        axes[2].set_ylabel('PnL')
        axes[2].grid(True)
        axes[2].legend()
        
        plt.tight_layout()
        
        # Distribution finale du PnL
        fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
        final_pnl = results['pnl'][:, -1]
        
        ax_hist.hist(final_pnl, bins=30, alpha=0.7, density=True)
        ax_hist.axvline(x=np.mean(final_pnl), color='r', linestyle='--', 
                        label=f'Moyenne: {np.mean(final_pnl):.4f}')
        
        ax_hist.set_title('Distribution du PnL final')
        ax_hist.set_xlabel('PnL')
        ax_hist.set_ylabel('Densité')
        ax_hist.grid(True)
        ax_hist.legend()
        
        plt.tight_layout()
        
        return (fig, axes), (fig_hist, ax_hist)