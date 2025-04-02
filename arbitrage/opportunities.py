import numpy as np

class ArbitrageAnalyzer:
    """
    Classe pour analyser les opportunités d'arbitrage entre obligations et dérivés de taux.
    """
    
    def __init__(self, bond_pricer, swap_pricer=None, option_pricer=None):
        """
        Initialise l'analyseur d'arbitrage.
        
        Args:
            bond_pricer: Pricer pour les obligations
            swap_pricer: Pricer pour les swaps de taux
            option_pricer: Pricer pour les options sur taux
        """
        self.bond_pricer = bond_pricer
        self.swap_pricer = swap_pricer
        self.option_pricer = option_pricer
        self.rate_model = bond_pricer.rate_model
    
    def analyze_bond_vs_swaps(self, bond_maturity, bond_coupon_rate, bond_price=None, frequency=1.0, 
                             notional=100.0, valuation_date=0.0, transaction_costs=0.0, spread_tolerance=0.0005):
        """
        Analyse les opportunités d'arbitrage entre une obligation et un swap de taux.
        
        Args:
            bond_maturity (float): Maturité de l'obligation en années
            bond_coupon_rate (float): Taux du coupon de l'obligation
            bond_price (float): Prix observé de l'obligation (si None, le prix théorique est utilisé)
            frequency (float): Fréquence des paiements de coupon par an
            notional (float): Valeur nominale
            valuation_date (float): Date d'évaluation
            transaction_costs (float): Coûts de transaction (en pourcentage)
            spread_tolerance (float): Tolérance pour les écarts de taux (en pourcentage)
            
        Returns:
            dict: Résultats de l'analyse d'arbitrage
        """
        if self.swap_pricer is None:
            raise ValueError("Un pricer de swap est nécessaire pour cette analyse")
        
        # Prix théorique de l'obligation
        theoretical_bond_price = self.bond_pricer.price_fixed_coupon_bond(
            bond_maturity, bond_coupon_rate, frequency, notional, valuation_date
        )
        
        # Prix observé de l'obligation (ou théorique si non fourni)
        observed_bond_price = bond_price if bond_price is not None else theoretical_bond_price
        
        # Yield to maturity de l'obligation observée
        bond_ytm = self.bond_pricer.calculate_yield_to_maturity(
            observed_bond_price, bond_maturity, bond_coupon_rate, frequency, notional, valuation_date
        )
        
        # Taux de swap à la parité pour la même maturité
        swap_rate = self.swap_pricer.par_rate(valuation_date, bond_maturity, frequency)
        
        # Écart entre le YTM de l'obligation et le taux de swap
        spread = bond_ytm - swap_rate
        
        # Opportunité d'arbitrage détectée si l'écart est supérieur au seuil de tolérance
        # en tenant compte des coûts de transaction
        arbitrage_opportunity = abs(spread) > (spread_tolerance + transaction_costs)
        
        # Direction de l'arbitrage
        if arbitrage_opportunity:
            if spread > 0:
                # YTM de l'obligation > taux de swap : acheter obligation, entrer dans swap payeur
                strategy = "Acheter l'obligation, entrer dans un swap payeur (taux fixe)"
            else:
                # YTM de l'obligation < taux de swap : vendre obligation, entrer dans swap receveur
                strategy = "Vendre l'obligation, entrer dans un swap receveur (taux fixe)"
        else:
            strategy = "Pas d'opportunité d'arbitrage significative"
        
        # Calcul du profit potentiel (approximatif)
        if arbitrage_opportunity:
            # Duration modifiée pour estimer la sensibilité
            modified_duration = self.bond_pricer.calculate_modified_duration(
                bond_maturity, bond_coupon_rate, frequency, notional, valuation_date
            )
            
            # Profit estimé pour un déplacement de courbe qui ramènerait le spread à zéro
            estimated_profit = abs(spread) * modified_duration * notional
            estimated_profit_after_costs = estimated_profit - (transaction_costs * notional)
        else:
            estimated_profit = 0.0
            estimated_profit_after_costs = 0.0
        
        # Résultat de l'analyse
        result = {
            "theoretical_bond_price": theoretical_bond_price,
            "observed_bond_price": observed_bond_price,
            "bond_ytm": bond_ytm,
            "swap_rate": swap_rate,
            "spread": spread,
            "arbitrage_opportunity": arbitrage_opportunity,
            "strategy": strategy,
            "estimated_profit": estimated_profit,
            "estimated_profit_after_costs": estimated_profit_after_costs
        }
        
        return result
    
    def analyze_bond_vs_capfloor(self, bond_maturity, bond_coupon_rate, cap_strike, floor_strike=None,
                                bond_price=None, frequency=1.0, notional=100.0, valuation_date=0.0,
                                volatility=0.015, transaction_costs=0.0, spread_tolerance=0.0005):
        """
        Analyse les opportunités d'arbitrage entre une obligation et des options sur taux (cap/floor).
        
        Args:
            bond_maturity (float): Maturité de l'obligation en années
            bond_coupon_rate (float): Taux du coupon de l'obligation
            cap_strike (float): Taux d'exercice du cap
            floor_strike (float): Taux d'exercice du floor (optionnel, pour les collars)
            bond_price (float): Prix observé de l'obligation (si None, le prix théorique est utilisé)
            frequency (float): Fréquence des paiements par an
            notional (float): Valeur nominale
            valuation_date (float): Date d'évaluation
            volatility (float): Volatilité pour le pricing des options
            transaction_costs (float): Coûts de transaction (en pourcentage)
            spread_tolerance (float): Tolérance pour les écarts de taux (en pourcentage)
            
        Returns:
            dict: Résultats de l'analyse d'arbitrage
        """
        if self.option_pricer is None:
            raise ValueError("Un pricer d'options sur taux est nécessaire pour cette analyse")
        
        # Prix théorique de l'obligation
        theoretical_bond_price = self.bond_pricer.price_fixed_coupon_bond(
            bond_maturity, bond_coupon_rate, frequency, notional, valuation_date
        )
        
        # Prix observé de l'obligation (ou théorique si non fourni)
        observed_bond_price = bond_price if bond_price is not None else theoretical_bond_price
        
        # Yield to maturity de l'obligation observée
        bond_ytm = self.bond_pricer.calculate_yield_to_maturity(
            observed_bond_price, bond_maturity, bond_coupon_rate, frequency, notional, valuation_date
        )
        
        # Taux forward moyens sur la période
        forward_rates = []
        for t in np.arange(valuation_date + frequency, bond_maturity + 1e-10, frequency):
            if t - frequency >= valuation_date:
                fwd = self.rate_model.forward_rate(valuation_date, t - frequency, t, self.rate_model.r0)
                forward_rates.append(fwd)
        
        avg_forward_rate = np.mean(forward_rates) if forward_rates else self.rate_model.r0
        
        # Prix du cap pour la maturité de l'obligation
        cap_price = self.option_pricer.price_cap(
            valuation_date=valuation_date,
            start_date=valuation_date,
            end_date=bond_maturity,
            strike=cap_strike,
            payment_frequency=frequency,
            notional=notional,
            volatility=volatility
        )
        
        # Prix du floor si fourni
        if floor_strike is not None:
            floor_price = self.option_pricer.price_floor(
                valuation_date=valuation_date,
                start_date=valuation_date,
                end_date=bond_maturity,
                strike=floor_strike,
                payment_frequency=frequency,
                notional=notional,
                volatility=volatility
            )
            collar_price = cap_price - floor_price
        else:
            floor_price = None
            collar_price = None
        
        # Analyse des opportunités d'arbitrage
        # La stratégie dépend des taux forward par rapport aux strikes des options
        
        if avg_forward_rate > cap_strike + spread_tolerance:
            # Taux forwards > strike du cap : acheter le cap est intéressant
            cap_strategy = "Acheter le cap"
            cap_arbitrage = True
        elif avg_forward_rate < cap_strike - spread_tolerance:
            # Taux forwards < strike du cap : vendre le cap est intéressant
            cap_strategy = "Vendre le cap"
            cap_arbitrage = True
        else:
            cap_strategy = "Pas d'opportunité claire avec le cap"
            cap_arbitrage = False
        
        if floor_strike is not None:
            if avg_forward_rate < floor_strike - spread_tolerance:
                # Taux forwards < strike du floor : acheter le floor est intéressant
                floor_strategy = "Acheter le floor"
                floor_arbitrage = True
            elif avg_forward_rate > floor_strike + spread_tolerance:
                # Taux forwards > strike du floor : vendre le floor est intéressant
                floor_strategy = "Vendre le floor"
                floor_arbitrage = True
            else:
                floor_strategy = "Pas d'opportunité claire avec le floor"
                floor_arbitrage = False
            
            # Pour le collar (long cap, short floor)
            if cap_arbitrage and floor_arbitrage:
                if avg_forward_rate > cap_strike and avg_forward_rate > floor_strike:
                    collar_strategy = "Acheter le cap, vendre le floor (collar)"
                    collar_arbitrage = True
                elif avg_forward_rate < cap_strike and avg_forward_rate < floor_strike:
                    collar_strategy = "Vendre le cap, acheter le floor (reverse collar)"
                    collar_arbitrage = True
                else:
                    collar_strategy = "Pas d'opportunité claire avec le collar"
                    collar_arbitrage = False
            else:
                collar_strategy = "Pas d'opportunité claire avec le collar"
                collar_arbitrage = False
        else:
            floor_strategy = "Floor non spécifié"
            floor_arbitrage = False
            collar_strategy = "Collar non applicable"
            collar_arbitrage = False
        
        # Résultat de l'analyse
        result = {
            "theoretical_bond_price": theoretical_bond_price,
            "observed_bond_price": observed_bond_price,
            "bond_ytm": bond_ytm,
            "avg_forward_rate": avg_forward_rate,
            "cap_strike": cap_strike,
            "cap_price": cap_price,
            "cap_strategy": cap_strategy,
            "cap_arbitrage": cap_arbitrage
        }
        
        if floor_strike is not None:
            result.update({
                "floor_strike": floor_strike,
                "floor_price": floor_price,
                "floor_strategy": floor_strategy,
                "floor_arbitrage": floor_arbitrage,
                "collar_price": collar_price,
                "collar_strategy": collar_strategy,
                "collar_arbitrage": collar_arbitrage
            })
        
        return result
    
    def analyze_asset_swap(self, bond_maturity, bond_coupon_rate, bond_price,
                          frequency=1.0, notional=100.0, valuation_date=0.0,
                          transaction_costs=0.0, spread_tolerance=0.0005):
        """
        Analyse les opportunités d'arbitrage d'asset swap (obligation + swap).
        
        Un asset swap combine l'achat d'une obligation et un swap de taux
        pour transformer les flux fixes de l'obligation en flux variables.
        
        Args:
            bond_maturity (float): Maturité de l'obligation en années
            bond_coupon_rate (float): Taux du coupon de l'obligation
            bond_price (float): Prix observé de l'obligation
            frequency (float): Fréquence des paiements par an
            notional (float): Valeur nominale
            valuation_date (float): Date d'évaluation
            transaction_costs (float): Coûts de transaction (en pourcentage)
            spread_tolerance (float): Tolérance pour les écarts de taux (en pourcentage)
            
        Returns:
            dict: Résultats de l'analyse d'arbitrage
        """
        if self.swap_pricer is None:
            raise ValueError("Un pricer de swap est nécessaire pour cette analyse")
        
        # Prix théorique de l'obligation
        theoretical_bond_price = self.bond_pricer.price_fixed_coupon_bond(
            bond_maturity, bond_coupon_rate, frequency, notional, valuation_date
        )
        
        # Yield to maturity de l'obligation observée
        bond_ytm = self.bond_pricer.calculate_yield_to_maturity(
            bond_price, bond_maturity, bond_coupon_rate, frequency, notional, valuation_date
        )
        
        # Taux de swap à la parité pour la même maturité
        swap_rate = self.swap_pricer.par_rate(valuation_date, bond_maturity, frequency)
        
        # Calcul du spread d'asset swap
        # Le spread d'asset swap est la différence entre le coupon de l'obligation
        # et le taux de swap, ajusté pour la différence entre le prix de l'obligation
        # et sa valeur nominale
        
        # Ajustement du coupon pour le prix de l'obligation
        adjusted_coupon = bond_coupon_rate * notional / bond_price
        
        # Spread d'asset swap
        asset_swap_spread = adjusted_coupon - swap_rate
        
        # Opportunité d'arbitrage détectée si l'écart est supérieur au seuil de tolérance
        # en tenant compte des coûts de transaction
        arbitrage_opportunity = asset_swap_spread > (spread_tolerance + transaction_costs)
        
        # Stratégie d'arbitrage
        if arbitrage_opportunity:
            strategy = "Acheter l'obligation, entrer dans un swap payeur (taux fixe)"
            
            # Profit estimé (simplifié)
            estimated_profit = asset_swap_spread * notional * bond_maturity
            estimated_profit_after_costs = estimated_profit - (transaction_costs * notional)
        else:
            strategy = "Pas d'opportunité d'arbitrage significative"
            estimated_profit = 0.0
            estimated_profit_after_costs = 0.0
        
        # Résultat de l'analyse
        result = {
            "theoretical_bond_price": theoretical_bond_price,
            "observed_bond_price": bond_price,
            "bond_ytm": bond_ytm,
            "swap_rate": swap_rate,
            "asset_swap_spread": asset_swap_spread,
            "arbitrage_opportunity": arbitrage_opportunity,
            "strategy": strategy,
            "estimated_profit": estimated_profit,
            "estimated_profit_after_costs": estimated_profit_after_costs
        }
        
        return result