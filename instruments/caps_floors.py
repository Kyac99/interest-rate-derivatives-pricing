import numpy as np
from datetime import datetime, timedelta

class Cap:
    """
    Classe représentant un Cap, qui est un ensemble d'options d'achat (caplets)
    sur un taux de référence.
    """
    
    def __init__(self, start_date, maturity_date, strike, payment_frequency=0.5, notional=1.0):
        """
        Initialise un Cap.
        
        Args:
            start_date: Date de début du Cap
            maturity_date: Date de maturité du Cap
            strike (float): Taux d'exercice
            payment_frequency (float): Fréquence des paiements en années
            notional (float): Montant notionnel
        """
        self.start_date = start_date
        self.maturity_date = maturity_date
        self.strike = strike
        self.payment_frequency = payment_frequency
        self.notional = notional
        
        # Générer les dates de paiement
        self.generate_payment_schedule()
    
    def generate_payment_schedule(self):
        """
        Génère le calendrier des paiements du Cap.
        """
        if isinstance(self.start_date, datetime) and isinstance(self.maturity_date, datetime):
            # Version avec les dates réelles
            frequency_days = int(self.payment_frequency * 365)
            current_date = self.start_date
            self.payment_dates = []
            
            while current_date < self.maturity_date:
                current_date += timedelta(days=frequency_days)
                if current_date <= self.maturity_date:
                    self.payment_dates.append(current_date)
        else:
            # Version simplifiée avec des nombres flottants représentant les années
            self.payment_dates = np.arange(
                self.start_date + self.payment_frequency,
                self.maturity_date + 1e-10,
                self.payment_frequency
            )
    
    def price(self, pricer, valuation_date=0, volatility=None):
        """
        Calcule le prix du Cap en utilisant un pricer donné.
        
        Args:
            pricer: Objet implémentant la méthode price_cap()
            valuation_date (float): Date d'évaluation
            volatility (float, optional): Volatilité utilisée pour le pricing
            
        Returns:
            float: Prix du Cap
        """
        return pricer.price_cap(
            valuation_date=valuation_date,
            start_date=self.start_date,
            end_date=self.maturity_date,
            strike=self.strike,
            payment_frequency=self.payment_frequency,
            notional=self.notional,
            volatility=volatility
        )
    
    def __str__(self):
        return f"Cap | Strike: {self.strike:.4f} | Notionnel: {self.notional:,.2f} | Maturité: {self.maturity_date}"


class Floor:
    """
    Classe représentant un Floor, qui est un ensemble d'options de vente (floorlets)
    sur un taux de référence.
    """
    
    def __init__(self, start_date, maturity_date, strike, payment_frequency=0.5, notional=1.0):
        """
        Initialise un Floor.
        
        Args:
            start_date: Date de début du Floor
            maturity_date: Date de maturité du Floor
            strike (float): Taux d'exercice
            payment_frequency (float): Fréquence des paiements en années
            notional (float): Montant notionnel
        """
        self.start_date = start_date
        self.maturity_date = maturity_date
        self.strike = strike
        self.payment_frequency = payment_frequency
        self.notional = notional
        
        # Générer les dates de paiement
        self.generate_payment_schedule()
    
    def generate_payment_schedule(self):
        """
        Génère le calendrier des paiements du Floor.
        """
        if isinstance(self.start_date, datetime) and isinstance(self.maturity_date, datetime):
            # Version avec les dates réelles
            frequency_days = int(self.payment_frequency * 365)
            current_date = self.start_date
            self.payment_dates = []
            
            while current_date < self.maturity_date:
                current_date += timedelta(days=frequency_days)
                if current_date <= self.maturity_date:
                    self.payment_dates.append(current_date)
        else:
            # Version simplifiée avec des nombres flottants représentant les années
            self.payment_dates = np.arange(
                self.start_date + self.payment_frequency,
                self.maturity_date + 1e-10,
                self.payment_frequency
            )
    
    def price(self, pricer, valuation_date=0, volatility=None):
        """
        Calcule le prix du Floor en utilisant un pricer donné.
        
        Args:
            pricer: Objet implémentant la méthode price_floor()
            valuation_date (float): Date d'évaluation
            volatility (float, optional): Volatilité utilisée pour le pricing
            
        Returns:
            float: Prix du Floor
        """
        return pricer.price_floor(
            valuation_date=valuation_date,
            start_date=self.start_date,
            end_date=self.maturity_date,
            strike=self.strike,
            payment_frequency=self.payment_frequency,
            notional=self.notional,
            volatility=volatility
        )
    
    def __str__(self):
        return f"Floor | Strike: {self.strike:.4f} | Notionnel: {self.notional:,.2f} | Maturité: {self.maturity_date}"


class Collar:
    """
    Classe représentant un Collar, qui est une combinaison d'un achat de Cap
    et d'une vente de Floor avec des strikes différents.
    """
    
    def __init__(self, start_date, maturity_date, cap_strike, floor_strike, payment_frequency=0.5, notional=1.0):
        """
        Initialise un Collar.
        
        Args:
            start_date: Date de début du Collar
            maturity_date: Date de maturité du Collar
            cap_strike (float): Taux d'exercice du Cap (acheté)
            floor_strike (float): Taux d'exercice du Floor (vendu)
            payment_frequency (float): Fréquence des paiements en années
            notional (float): Montant notionnel
        """
        self.cap = Cap(start_date, maturity_date, cap_strike, payment_frequency, notional)
        self.floor = Floor(start_date, maturity_date, floor_strike, payment_frequency, notional)
    
    def price(self, pricer, valuation_date=0, volatility=None):
        """
        Calcule le prix du Collar en utilisant un pricer donné.
        
        Args:
            pricer: Objet implémentant les méthodes price_cap() et price_floor()
            valuation_date (float): Date d'évaluation
            volatility (float, optional): Volatilité utilisée pour le pricing
            
        Returns:
            float: Prix du Collar
        """
        cap_price = self.cap.price(pricer, valuation_date, volatility)
        floor_price = self.floor.price(pricer, valuation_date, volatility)
        
        # Long cap, short floor
        collar_price = cap_price - floor_price
        
        return collar_price
    
    def __str__(self):
        return f"Collar | Cap Strike: {self.cap.strike:.4f} | Floor Strike: {self.floor.strike:.4f} | Notionnel: {self.cap.notional:,.2f} | Maturité: {self.cap.maturity_date}"