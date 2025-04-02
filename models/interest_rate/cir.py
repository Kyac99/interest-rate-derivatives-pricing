import numpy as np
from scipy.special import gamma
from .base_model import InterestRateModel

class CIR(InterestRateModel):
    """
    Implémentation du modèle de Cox-Ingersoll-Ross (CIR):
    dr(t) = kappa * (theta - r(t)) * dt + sigma * sqrt(r(t)) * dW(t)
    
    où:
    - kappa: vitesse de retour à la moyenne
    - theta: niveau moyen à long terme
    - sigma: volatilité
    
    Contrairement au modèle de Vasicek, le modèle CIR garantit que les taux d'intérêt
    restent positifs, car la volatilité est proportionnelle à la racine carrée du taux.
    """
    
    def __init__(self, r0, kappa, theta, sigma, timesteps=100, time_horizon=1.0):
        """
        Initialise le modèle CIR.
        
        Args:
            r0 (float): Taux d'intérêt initial
            kappa (float): Vitesse de retour à la moyenne
            theta (float): Niveau moyen à long terme
            sigma (float): Volatilité
            timesteps (int): Nombre de pas de temps pour la simulation
            time_horizon (float): Horizon temporel en années
        """
        if r0 <= 0:
            raise ValueError("Le taux initial r0 doit être strictement positif")
        
        if 2 * kappa * theta <= sigma**2:
            raise ValueError("La condition de Feller (2*kappa*theta > sigma^2) n'est pas respectée")
        
        params = {'kappa': kappa, 'theta': theta, 'sigma': sigma}
        super().__init__(r0, params, timesteps, time_horizon)
        
    def simulate_rates(self, n_paths=1, seed=None):
        """
        Simule les trajectoires de taux selon le modèle CIR.
        
        Args:
            n_paths (int): Nombre de trajectoires à simuler
            seed (int, optional): Graine pour la génération aléatoire
            
        Returns:
            numpy.ndarray: Matrice des taux simulés de forme (n_paths, timesteps+1)
        """
        if seed is not None:
            np.random.seed(seed)
            
        kappa = self.params['kappa']
        theta = self.params['theta']
        sigma = self.params['sigma']
        dt = self.dt
        
        # Initialisation des trajectoires
        rates = np.zeros((n_paths, self.timesteps + 1))
        rates[:, 0] = self.r0
        
        # Schéma d'Euler modifié pour garantir des taux positifs
        for t in range(1, self.timesteps + 1):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            
            # Utilisation du schéma de discrétisation non-central chi-squared
            # Pour garantir la positivité des taux
            d = 4 * kappa * theta / sigma**2
            c = sigma**2 * (1 - np.exp(-kappa * dt)) / (4 * kappa)
            ncp = 4 * kappa * np.exp(-kappa * dt) / (sigma**2 * (1 - np.exp(-kappa * dt))) * rates[:, t-1]
            
            # On peut aussi utiliser une approximation d'Euler avec une réflexion en zéro
            drift = kappa * (theta - np.maximum(0, rates[:, t-1])) * dt
            diffusion = sigma * np.sqrt(np.maximum(0, rates[:, t-1]) * dt) * dW
            rates[:, t] = np.maximum(0, rates[:, t-1] + drift + diffusion)
            
        return rates
    
    def zero_coupon_bond_price(self, t, T, r):
        """
        Calcule le prix d'une obligation zéro-coupon selon le modèle CIR.
        Formule analytique disponible dans ce cas.
        
        Args:
            t (float): Date actuelle
            T (float): Date de maturité
            r (float): Taux d'intérêt actuel
            
        Returns:
            float: Prix de l'obligation zéro-coupon
        """
        if t >= T:
            return 1.0
            
        kappa = self.params['kappa']
        theta = self.params['theta']
        sigma = self.params['sigma']
        
        h = np.sqrt(kappa**2 + 2 * sigma**2)
        tau = T - t
        
        # Paramètre A de la formule CIR
        A_num = 2 * h * np.exp((kappa + h) * tau / 2)
        A_denom = 2 * h + (kappa + h) * (np.exp(h * tau) - 1)
        A = (A_num / A_denom) ** (2 * kappa * theta / sigma**2)
        
        # Paramètre B de la formule CIR
        B_num = 2 * (np.exp(h * tau) - 1)
        B_denom = 2 * h + (kappa + h) * (np.exp(h * tau) - 1)
        B = B_num / B_denom
        
        # Prix de l'obligation
        return A * np.exp(-B * r)