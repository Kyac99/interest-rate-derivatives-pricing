import numpy as np
from datetime import datetime, timedelta
from .swaps import InterestRateSwap

class Swaption:
    """
    Classe représentant une option sur swap (swaption).
    
    Une swaption donne le droit (mais non l'obligation) de rentrer dans un swap
    de taux à une date future avec un taux fixé à l'avance.
    """
    
    def __init__(self, expiry_date, underlying_swap_maturity, strike_rate, is_payer=True, 
                 payment_frequency=0.5, notional=1.0):
        """
        Initialise une swaption.
        
        Args:
            expiry_date: Date d'expiration de la swaption
            underlying_swap_maturity: Maturité du swap sous-jacent
            strike_rate (float): Taux fixe du swap sous-jacent
            is_payer (bool): True pour une swaption payeuse, False pour une receveuse
            payment_frequency (float): Fréquence des paiements du swap sous-jacent en années
            notional (float): Montant notionnel
        """
        self.expiry_date = expiry_date
        self.underlying_swap_maturity = underlying_swap_maturity
        self.strike_rate = strike_rate
        self.is_payer = is_payer
        self.payment_frequency = payment_frequency
        self.notional = notional
        
        # Le swap sous-jacent commence à la date d'expiration de la swaption
        self.underlying_swap_start = self.expiry_date
        self.underlying_swap_end = self.underlying_swap_start + self.underlying_swap_maturity
        
        # Création du swap sous-jacent
        self.underlying_swap = InterestRateSwap(
            start_date=self.underlying_swap_start,
            maturity_date=self.underlying_swap_end,
            fixed_rate=self.strike_rate,
            payment_frequency=self.payment_frequency,
            notional=self.notional,
            is_payer=self.is_payer
        )
    
    def price(self, pricer, valuation_date=0, volatility=None):
        """
        Calcule le prix de la swaption en utilisant un pricer donné.
        
        Args:
            pricer: Objet implémentant la méthode price() pour les swaptions
            valuation_date (float): Date d'évaluation
            volatility (float, optional): Volatilité utilisée pour le pricing
            
        Returns:
            float: Prix de la swaption
        """
        return pricer.price(
            valuation_date=valuation_date,
            expiry_date=self.expiry_date,
            underlying_swap_start=self.underlying_swap_start,
            underlying_swap_end=self.underlying_swap_end,
            strike=self.strike_rate,
            payment_frequency=self.payment_frequency,
            notional=self.notional,
            volatility=volatility,
            is_payer=self.is_payer
        )
    
    def __str__(self):
        swaption_type = "Payeuse" if self.is_payer else "Receveuse"
        return f"Swaption {swaption_type} | Strike: {self.strike_rate:.4f} | Expiration: {self.expiry_date} | Maturité swap: {self.underlying_swap_maturity} ans | Notionnel: {self.notional:,.2f}"