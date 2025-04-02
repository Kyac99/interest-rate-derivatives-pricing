import numpy as np

class BondPricer:
    """
    Classe pour le pricing d'obligations.
    """
    
    def __init__(self, rate_model, credit_spread=0.0):
        """
        Initialise le pricer d'obligations.
        
        Args:
            rate_model: Modèle de taux d'intérêt
            credit_spread (float): Spread de crédit à appliquer (en décimal)
        """
        self.rate_model = rate_model
        self.credit_spread = credit_spread
    
    def price_zero_coupon_bond(self, maturity, notional=100.0, valuation_date=0.0):
        """
        Calcule le prix d'une obligation zéro-coupon.
        
        Args:
            maturity (float): Maturité de l'obligation en années
            notional (float): Valeur nominale de l'obligation
            valuation_date (float): Date d'évaluation
            
        Returns:
            float: Prix de l'obligation zéro-coupon
        """
        if maturity <= valuation_date:
            return notional
        
        current_rate = self.rate_model.r0
        
        # Facteur d'actualisation du modèle de taux
        discount_factor = self.rate_model.zero_coupon_bond_price(
            valuation_date, maturity, current_rate
        )
        
        # Application du spread de crédit
        if self.credit_spread > 0:
            time_to_maturity = maturity - valuation_date
            credit_discount = np.exp(-self.credit_spread * time_to_maturity)
            discount_factor *= credit_discount
        
        # Prix de l'obligation
        bond_price = notional * discount_factor
        
        return bond_price
    
    def price_fixed_coupon_bond(self, maturity, coupon_rate, frequency=1.0, notional=100.0, valuation_date=0.0):
        """
        Calcule le prix d'une obligation à coupon fixe.
        
        Args:
            maturity (float): Maturité de l'obligation en années
            coupon_rate (float): Taux du coupon annuel (en décimal)
            frequency (float): Fréquence des paiements de coupon par an
            notional (float): Valeur nominale de l'obligation
            valuation_date (float): Date d'évaluation
            
        Returns:
            float: Prix de l'obligation à coupon fixe
        """
        if maturity <= valuation_date:
            return notional
        
        current_rate = self.rate_model.r0
        
        # Calcul des dates de paiement des coupons
        coupon_dates = np.arange(
            valuation_date + frequency,
            maturity + 1e-10,
            frequency
        )
        
        # Si valuation_date tombe entre deux dates de coupon
        if valuation_date > 0:
            original_coupon_dates = np.arange(
                frequency,
                maturity + 1e-10,
                frequency
            )
            coupon_dates = original_coupon_dates[original_coupon_dates > valuation_date]
        
        # Montant du coupon
        coupon_amount = notional * coupon_rate * frequency
        
        # Prix de l'obligation comme somme des flux actualisés
        bond_price = 0.0
        
        # Actualisation des coupons
        for coupon_date in coupon_dates:
            # Facteur d'actualisation du modèle de taux
            discount_factor = self.rate_model.zero_coupon_bond_price(
                valuation_date, coupon_date, current_rate
            )
            
            # Application du spread de crédit
            if self.credit_spread > 0:
                time_to_payment = coupon_date - valuation_date
                credit_discount = np.exp(-self.credit_spread * time_to_payment)
                discount_factor *= credit_discount
            
            # Ajout du coupon actualisé
            bond_price += coupon_amount * discount_factor
        
        # Actualisation du principal (remboursement à maturité)
        discount_factor = self.rate_model.zero_coupon_bond_price(
            valuation_date, maturity, current_rate
        )
        
        # Application du spread de crédit pour le principal
        if self.credit_spread > 0:
            time_to_maturity = maturity - valuation_date
            credit_discount = np.exp(-self.credit_spread * time_to_maturity)
            discount_factor *= credit_discount
        
        # Ajout du principal actualisé
        bond_price += notional * discount_factor
        
        return bond_price
    
    def calculate_yield_to_maturity(self, bond_price, maturity, coupon_rate, frequency=1.0, notional=100.0, valuation_date=0.0, max_iterations=100, precision=1e-8):
        """
        Calcule le rendement à maturité (YTM) d'une obligation.
        
        Args:
            bond_price (float): Prix observé de l'obligation
            maturity (float): Maturité de l'obligation en années
            coupon_rate (float): Taux du coupon annuel (en décimal)
            frequency (float): Fréquence des paiements de coupon par an
            notional (float): Valeur nominale de l'obligation
            valuation_date (float): Date d'évaluation
            max_iterations (int): Nombre maximum d'itérations pour la recherche
            precision (float): Précision souhaitée pour le résultat
            
        Returns:
            float: Rendement à maturité (YTM) de l'obligation
        """
        if maturity <= valuation_date:
            return 0.0
        
        # Fonction pour calculer la différence entre le prix calculé et le prix observé
        def price_difference(ytm):
            # Calcul du prix avec un taux d'actualisation constant égal au YTM
            price = 0.0
            
            # Calcul des dates de paiement des coupons
            coupon_dates = np.arange(
                valuation_date + frequency,
                maturity + 1e-10,
                frequency
            )
            
            # Si valuation_date tombe entre deux dates de coupon
            if valuation_date > 0:
                original_coupon_dates = np.arange(
                    frequency,
                    maturity + 1e-10,
                    frequency
                )
                coupon_dates = original_coupon_dates[original_coupon_dates > valuation_date]
            
            # Montant du coupon
            coupon_amount = notional * coupon_rate * frequency
            
            # Actualisation des coupons
            for coupon_date in coupon_dates:
                time_to_payment = coupon_date - valuation_date
                discount_factor = np.exp(-ytm * time_to_payment)
                price += coupon_amount * discount_factor
            
            # Actualisation du principal
            time_to_maturity = maturity - valuation_date
            discount_factor = np.exp(-ytm * time_to_maturity)
            price += notional * discount_factor
            
            return price - bond_price
        
        # Recherche du YTM par la méthode de la bissection
        low_rate = 0.0001  # 0.01%
        high_rate = 0.5    # 50%
        
        for _ in range(max_iterations):
            mid_rate = (low_rate + high_rate) / 2.0
            
            if abs(price_difference(mid_rate)) < precision:
                return mid_rate
            
            if price_difference(mid_rate) * price_difference(low_rate) < 0:
                high_rate = mid_rate
            else:
                low_rate = mid_rate
        
        # Si la précision n'est pas atteinte, retourne la meilleure approximation
        return (low_rate + high_rate) / 2.0
    
    def calculate_duration(self, maturity, coupon_rate, frequency=1.0, notional=100.0, valuation_date=0.0):
        """
        Calcule la duration de Macaulay d'une obligation.
        
        Args:
            maturity (float): Maturité de l'obligation en années
            coupon_rate (float): Taux du coupon annuel (en décimal)
            frequency (float): Fréquence des paiements de coupon par an
            notional (float): Valeur nominale de l'obligation
            valuation_date (float): Date d'évaluation
            
        Returns:
            float: Duration de Macaulay en années
        """
        if maturity <= valuation_date:
            return 0.0
        
        # Prix de l'obligation
        bond_price = self.price_fixed_coupon_bond(
            maturity, coupon_rate, frequency, notional, valuation_date
        )
        
        # Yield to maturity
        ytm = self.calculate_yield_to_maturity(
            bond_price, maturity, coupon_rate, frequency, notional, valuation_date
        )
        
        # Calcul des dates de paiement des coupons
        coupon_dates = np.arange(
            valuation_date + frequency,
            maturity + 1e-10,
            frequency
        )
        
        # Si valuation_date tombe entre deux dates de coupon
        if valuation_date > 0:
            original_coupon_dates = np.arange(
                frequency,
                maturity + 1e-10,
                frequency
            )
            coupon_dates = original_coupon_dates[original_coupon_dates > valuation_date]
        
        # Montant du coupon
        coupon_amount = notional * coupon_rate * frequency
        
        # Calcul de la duration
        weighted_time_sum = 0.0
        
        # Contribution des coupons
        for coupon_date in coupon_dates:
            time_to_payment = coupon_date - valuation_date
            discount_factor = np.exp(-ytm * time_to_payment)
            present_value = coupon_amount * discount_factor
            weighted_time_sum += time_to_payment * present_value
        
        # Contribution du principal
        time_to_maturity = maturity - valuation_date
        discount_factor = np.exp(-ytm * time_to_maturity)
        present_value = notional * discount_factor
        weighted_time_sum += time_to_maturity * present_value
        
        # Duration
        duration = weighted_time_sum / bond_price
        
        return duration
    
    def calculate_modified_duration(self, maturity, coupon_rate, frequency=1.0, notional=100.0, valuation_date=0.0):
        """
        Calcule la duration modifiée d'une obligation.
        
        Args:
            maturity (float): Maturité de l'obligation en années
            coupon_rate (float): Taux du coupon annuel (en décimal)
            frequency (float): Fréquence des paiements de coupon par an
            notional (float): Valeur nominale de l'obligation
            valuation_date (float): Date d'évaluation
            
        Returns:
            float: Duration modifiée
        """
        # Prix de l'obligation
        bond_price = self.price_fixed_coupon_bond(
            maturity, coupon_rate, frequency, notional, valuation_date
        )
        
        # Yield to maturity
        ytm = self.calculate_yield_to_maturity(
            bond_price, maturity, coupon_rate, frequency, notional, valuation_date
        )
        
        # Duration de Macaulay
        macaulay_duration = self.calculate_duration(
            maturity, coupon_rate, frequency, notional, valuation_date
        )
        
        # Duration modifiée
        modified_duration = macaulay_duration / (1 + ytm / frequency)
        
        return modified_duration