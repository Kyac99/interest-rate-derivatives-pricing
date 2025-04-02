import numpy as np
from scipy.stats import norm

class CapFloorPricer:
    """
    Pricer pour les Caps et Floors sur taux d'intérêt.
    
    Un Cap est un ensemble d'options d'achat (caplets) sur un taux de référence.
    Un Floor est un ensemble d'options de vente (floorlets) sur un taux de référence.
    """
    
    def __init__(self, rate_model, volatility_model=None):
        """
        Initialise le pricer de Caps et Floors.
        
        Args:
            rate_model: Modèle de taux d'intérêt
            volatility_model: Modèle de volatilité (optionnel)
        """
        self.rate_model = rate_model
        self.volatility_model = volatility_model
    
    def black_price_caplet(self, forward_rate, strike, time_to_maturity, volatility, discount_factor, notional=1.0, delta_t=0.5):
        """
        Calcule le prix d'un caplet en utilisant la formule de Black.
        
        Args:
            forward_rate (float): Taux forward
            strike (float): Taux d'exercice
            time_to_maturity (float): Temps jusqu'à la maturité en années
            volatility (float): Volatilité du taux
            discount_factor (float): Facteur d'actualisation
            notional (float): Montant notionnel
            delta_t (float): Période de fixation du taux
            
        Returns:
            float: Prix du caplet
        """
        if time_to_maturity <= 0:
            # Option expirée ou à maturité
            return max(0, forward_rate - strike) * discount_factor * notional * delta_t
        
        # Prix Black-Scholes avec taux forward comme sous-jacent
        d1 = (np.log(forward_rate / strike) + (volatility**2 / 2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
        d2 = d1 - volatility * np.sqrt(time_to_maturity)
        
        price = discount_factor * notional * delta_t * (forward_rate * norm.cdf(d1) - strike * norm.cdf(d2))
        return price
    
    def black_price_floorlet(self, forward_rate, strike, time_to_maturity, volatility, discount_factor, notional=1.0, delta_t=0.5):
        """
        Calcule le prix d'un floorlet en utilisant la formule de Black.
        
        Args:
            forward_rate (float): Taux forward
            strike (float): Taux d'exercice
            time_to_maturity (float): Temps jusqu'à la maturité en années
            volatility (float): Volatilité du taux
            discount_factor (float): Facteur d'actualisation
            notional (float): Montant notionnel
            delta_t (float): Période de fixation du taux
            
        Returns:
            float: Prix du floorlet
        """
        if time_to_maturity <= 0:
            # Option expirée ou à maturité
            return max(0, strike - forward_rate) * discount_factor * notional * delta_t
        
        # Prix Black-Scholes avec taux forward comme sous-jacent
        d1 = (np.log(forward_rate / strike) + (volatility**2 / 2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
        d2 = d1 - volatility * np.sqrt(time_to_maturity)
        
        price = discount_factor * notional * delta_t * (strike * norm.cdf(-d2) - forward_rate * norm.cdf(-d1))
        return price
    
    def price_cap(self, valuation_date, start_date, end_date, strike, payment_frequency=0.5, notional=1.0, volatility=None):
        """
        Calcule le prix d'un Cap.
        
        Args:
            valuation_date (float): Date d'évaluation
            start_date (float): Date de début du Cap
            end_date (float): Date de fin du Cap
            strike (float): Taux d'exercice
            payment_frequency (float): Fréquence des paiements en années
            notional (float): Montant notionnel
            volatility (float, optional): Volatilité utilisée pour le pricing
            
        Returns:
            float: Prix du Cap
        """
        # Taux actuel
        current_rate = self.rate_model.r0
        
        # Calcul des dates de fixation/paiement
        payment_dates = np.arange(start_date + payment_frequency, end_date + 1e-10, payment_frequency)
        
        # Prix total du Cap (somme des caplets)
        cap_price = 0.0
        
        for idx, payment_date in enumerate(payment_dates):
            # Période de fixation
            if idx == 0:
                fixing_start = start_date
            else:
                fixing_start = payment_dates[idx-1]
            
            fixing_end = payment_date
            
            # Taux forward pour la période
            forward_rate = self.rate_model.forward_rate(valuation_date, fixing_start, fixing_end, current_rate)
            
            # Facteur d'actualisation
            discount_factor = self.rate_model.zero_coupon_bond_price(valuation_date, payment_date, current_rate)
            
            # Temps jusqu'à la fixation
            time_to_fixing = max(0, fixing_start - valuation_date)
            
            # Volatilité (constante ou provenant d'un modèle)
            vol = volatility if volatility is not None else 0.2  # Valeur par défaut
            
            # Prix du caplet
            caplet_price = self.black_price_caplet(
                forward_rate=forward_rate,
                strike=strike,
                time_to_maturity=time_to_fixing,
                volatility=vol,
                discount_factor=discount_factor,
                notional=notional,
                delta_t=payment_frequency
            )
            
            cap_price += caplet_price
        
        return cap_price
    
    def price_floor(self, valuation_date, start_date, end_date, strike, payment_frequency=0.5, notional=1.0, volatility=None):
        """
        Calcule le prix d'un Floor.
        
        Args:
            valuation_date (float): Date d'évaluation
            start_date (float): Date de début du Floor
            end_date (float): Date de fin du Floor
            strike (float): Taux d'exercice
            payment_frequency (float): Fréquence des paiements en années
            notional (float): Montant notionnel
            volatility (float, optional): Volatilité utilisée pour le pricing
            
        Returns:
            float: Prix du Floor
        """
        # Taux actuel
        current_rate = self.rate_model.r0
        
        # Calcul des dates de fixation/paiement
        payment_dates = np.arange(start_date + payment_frequency, end_date + 1e-10, payment_frequency)
        
        # Prix total du Floor (somme des floorlets)
        floor_price = 0.0
        
        for idx, payment_date in enumerate(payment_dates):
            # Période de fixation
            if idx == 0:
                fixing_start = start_date
            else:
                fixing_start = payment_dates[idx-1]
            
            fixing_end = payment_date
            
            # Taux forward pour la période
            forward_rate = self.rate_model.forward_rate(valuation_date, fixing_start, fixing_end, current_rate)
            
            # Facteur d'actualisation
            discount_factor = self.rate_model.zero_coupon_bond_price(valuation_date, payment_date, current_rate)
            
            # Temps jusqu'à la fixation
            time_to_fixing = max(0, fixing_start - valuation_date)
            
            # Volatilité (constante ou provenant d'un modèle)
            vol = volatility if volatility is not None else 0.2  # Valeur par défaut
            
            # Prix du floorlet
            floorlet_price = self.black_price_floorlet(
                forward_rate=forward_rate,
                strike=strike,
                time_to_maturity=time_to_fixing,
                volatility=vol,
                discount_factor=discount_factor,
                notional=notional,
                delta_t=payment_frequency
            )
            
            floor_price += floorlet_price
        
        return floor_price


class SwaptionPricer:
    """
    Pricer pour les options sur swap (swaptions).
    
    Une swaption donne le droit (mais non l'obligation) de rentrer dans un swap
    de taux à une date future avec un taux fixé à l'avance.
    """
    
    def __init__(self, rate_model, swap_pricer, volatility_model=None):
        """
        Initialise le pricer de swaptions.
        
        Args:
            rate_model: Modèle de taux d'intérêt
            swap_pricer: Pricer pour les swaps de taux
            volatility_model: Modèle de volatilité des taux de swap (optionnel)
        """
        self.rate_model = rate_model
        self.swap_pricer = swap_pricer
        self.volatility_model = volatility_model
    
    def black_price(self, forward_swap_rate, strike, swap_annuity, time_to_expiry, volatility, is_payer=True):
        """
        Calcule le prix d'une swaption en utilisant la formule de Black.
        
        Args:
            forward_swap_rate (float): Taux de swap forward
            strike (float): Taux fixe du swap sous-jacent
            swap_annuity (float): Annuité du swap sous-jacent
            time_to_expiry (float): Temps jusqu'à l'expiration en années
            volatility (float): Volatilité du taux de swap
            is_payer (bool): True pour une swaption payeuse, False pour une receveuse
            
        Returns:
            float: Prix de la swaption
        """
        if time_to_expiry <= 0:
            # Swaption expirée
            if is_payer:
                return max(0, forward_swap_rate - strike) * swap_annuity
            else:
                return max(0, strike - forward_swap_rate) * swap_annuity
        
        # Prix Black-Scholes
        d1 = (np.log(forward_swap_rate / strike) + (volatility**2 / 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        if is_payer:
            # Swaption payeuse (call sur le taux de swap)
            price = swap_annuity * (forward_swap_rate * norm.cdf(d1) - strike * norm.cdf(d2))
        else:
            # Swaption receveuse (put sur le taux de swap)
            price = swap_annuity * (strike * norm.cdf(-d2) - forward_swap_rate * norm.cdf(-d1))
        
        return price
    
    def price(self, valuation_date, expiry_date, underlying_swap_start, underlying_swap_end, strike, 
              payment_frequency=0.5, notional=1.0, volatility=None, is_payer=True):
        """
        Calcule le prix d'une swaption.
        
        Args:
            valuation_date (float): Date d'évaluation
            expiry_date (float): Date d'expiration de la swaption
            underlying_swap_start (float): Date de début du swap sous-jacent (= expiry_date)
            underlying_swap_end (float): Date de fin du swap sous-jacent
            strike (float): Taux fixe du swap sous-jacent
            payment_frequency (float): Fréquence des paiements du swap sous-jacent en années
            notional (float): Montant notionnel
            volatility (float, optional): Volatilité utilisée pour le pricing
            is_payer (bool): True pour une swaption payeuse, False pour une receveuse
            
        Returns:
            float: Prix de la swaption
        """
        # Vérification de la date d'expiration
        if expiry_date < underlying_swap_start:
            raise ValueError("La date d'expiration doit être égale à la date de début du swap sous-jacent")
        
        # Taux actuel
        current_rate = self.rate_model.r0
        
        # Temps jusqu'à l'expiration
        time_to_expiry = max(0, expiry_date - valuation_date)
        
        # Calcul du taux forward de swap
        forward_swap_rate = self.swap_pricer.par_rate(
            underlying_swap_start, 
            underlying_swap_end, 
            payment_frequency
        )
        
        # Calcul de l'annuité du swap
        payment_dates = np.arange(underlying_swap_start + payment_frequency, underlying_swap_end + 1e-10, payment_frequency)
        
        swap_annuity = 0.0
        for payment_date in payment_dates:
            discount_factor = self.rate_model.zero_coupon_bond_price(
                valuation_date, 
                payment_date, 
                current_rate
            )
            swap_annuity += payment_frequency * discount_factor * notional
        
        # Volatilité (constante ou provenant d'un modèle)
        vol = volatility if volatility is not None else 0.2  # Valeur par défaut
        
        # Calcul du prix de la swaption
        swaption_price = self.black_price(
            forward_swap_rate=forward_swap_rate,
            strike=strike,
            swap_annuity=swap_annuity,
            time_to_expiry=time_to_expiry,
            volatility=vol,
            is_payer=is_payer
        )
        
        return swaption_price