import numpy as np
import matplotlib.pyplot as plt

# Import des modules nécessaires du projet
from models.interest_rate import Vasicek
from models.derivatives import IRSwapPricer, CapFloorPricer
from arbitrage.bond_pricing import BondPricer
from arbitrage.opportunities import ArbitrageAnalyzer

def main():
    # Paramètres du modèle de Vasicek
    r0 = 0.03       # Taux initial
    kappa = 0.5     # Vitesse de retour à la moyenne
    theta = 0.05    # Niveau moyen à long terme
    sigma = 0.01    # Volatilité
    
    # Création du modèle de taux
    vasicek_model = Vasicek(r0, kappa, theta, sigma)
    
    # Création des pricers
    bond_pricer = BondPricer(vasicek_model)
    swap_pricer = IRSwapPricer(vasicek_model)
    capfloor_pricer = CapFloorPricer(vasicek_model)
    
    # Création de l'analyseur d'arbitrage
    arbitrage_analyzer = ArbitrageAnalyzer(
        bond_pricer=bond_pricer,
        swap_pricer=swap_pricer,
        option_pricer=capfloor_pricer
    )
    
    print("=" * 60)
    print("ANALYSE D'ARBITRAGE ENTRE OBLIGATIONS ET DÉRIVÉS DE TAUX")
    print("=" * 60)
    
    # ==============================
    # 1. Analyse Obligation vs Swap
    # ==============================
    print("\n1. ANALYSE OBLIGATION VS SWAP")
    print("-" * 40)
    
    # Paramètres de l'obligation
    bond_maturity = 5.0    # Maturité de 5 ans
    bond_coupon = 0.04     # Coupon de 4%
    
    # Prix théorique
    bond_price_theoretical = bond_pricer.price_fixed_coupon_bond(
        maturity=bond_maturity,
        coupon_rate=bond_coupon
    )
    
    # Scénarios de prix observés (prix théorique avec écarts)
    bond_prices = [
        bond_price_theoretical * 0.98,  # Obligation sous-évaluée (-2%)
        bond_price_theoretical,         # Prix théorique
        bond_price_theoretical * 1.02   # Obligation surévaluée (+2%)
    ]
    
    price_scenarios = ["Sous-évalué (-2%)", "Prix théorique", "Surévalué (+2%)"]
    
    # Analyse pour chaque scénario de prix
    for i, bond_price in enumerate(bond_prices):
        print(f"\nScénario: {price_scenarios[i]}")
        print(f"Prix de l'obligation: {bond_price:.4f}")
        
        # Analyse d'arbitrage
        result = arbitrage_analyzer.analyze_bond_vs_swaps(
            bond_maturity=bond_maturity,
            bond_coupon_rate=bond_coupon,
            bond_price=bond_price,
            transaction_costs=0.001  # 0.1% de coûts de transaction
        )
        
        # Affichage des résultats
        print(f"YTM de l'obligation: {result['bond_ytm']:.4%}")
        print(f"Taux de swap: {result['swap_rate']:.4%}")
        print(f"Spread: {result['spread']:.4%}")
        print(f"Opportunité d'arbitrage: {'Oui' if result['arbitrage_opportunity'] else 'Non'}")
        if result['arbitrage_opportunity']:
            print(f"Stratégie: {result['strategy']}")
            print(f"Profit estimé (après coûts): {result['estimated_profit_after_costs']:.4f}")
    
    # ==============================
    # 2. Analyse Obligation vs Cap/Floor
    # ==============================
    print("\n\n2. ANALYSE OBLIGATION VS CAP/FLOOR")
    print("-" * 40)
    
    # Paramètres des options
    cap_strike = 0.045    # Strike du Cap à 4.5%
    floor_strike = 0.025  # Strike du Floor à 2.5%
    volatility = 0.015    # Volatilité de 1.5%
    
    # Analyse pour le scénario de prix théorique
    print("\nAnalyse avec Cap et Floor:")
    
    # Analyse d'arbitrage avec les options
    result = arbitrage_analyzer.analyze_bond_vs_capfloor(
        bond_maturity=bond_maturity,
        bond_coupon_rate=bond_coupon,
        cap_strike=cap_strike,
        floor_strike=floor_strike,
        bond_price=bond_price_theoretical,
        volatility=volatility,
        transaction_costs=0.002  # 0.2% de coûts de transaction (plus élevés pour les options)
    )
    
    # Affichage des résultats
    print(f"Prix de l'obligation: {result['observed_bond_price']:.4f}")
    print(f"YTM de l'obligation: {result['bond_ytm']:.4%}")
    print(f"Taux forward moyen: {result['avg_forward_rate']:.4%}")
    print(f"\nPrix du Cap (Strike = {cap_strike:.2%}): {result['cap_price']:.4f}")
    print(f"Stratégie (Cap): {result['cap_strategy']}")
    print(f"\nPrix du Floor (Strike = {floor_strike:.2%}): {result['floor_price']:.4f}")
    print(f"Stratégie (Floor): {result['floor_strategy']}")
    print(f"\nPrix du Collar: {result['collar_price']:.4f}")
    print(f"Stratégie (Collar): {result['collar_strategy']}")
    
    # ==============================
    # 3. Analyse d'Asset Swap
    # ==============================
    print("\n\n3. ANALYSE D'ASSET SWAP")
    print("-" * 40)
    
    # Paramètres de l'obligation
    bond_maturity = 5.0    # Maturité de 5 ans
    bond_coupon = 0.05     # Coupon de 5% (plus élevé que le taux de swap)
    
    # Prix théorique
    bond_price_theoretical = bond_pricer.price_fixed_coupon_bond(
        maturity=bond_maturity,
        coupon_rate=bond_coupon
    )
    
    # Scénario de prix avec décote
    bond_price_discounted = bond_price_theoretical * 0.97  # Obligation avec décote (-3%)
    
    # Analyse d'asset swap
    result = arbitrage_analyzer.analyze_asset_swap(
        bond_maturity=bond_maturity,
        bond_coupon_rate=bond_coupon,
        bond_price=bond_price_discounted,
        transaction_costs=0.001  # 0.1% de coûts de transaction
    )
    
    # Affichage des résultats
    print(f"Prix théorique de l'obligation: {result['theoretical_bond_price']:.4f}")
    print(f"Prix observé de l'obligation: {result['observed_bond_price']:.4f}")
    print(f"YTM de l'obligation: {result['bond_ytm']:.4%}")
    print(f"Taux de swap: {result['swap_rate']:.4%}")
    print(f"Spread d'asset swap: {result['asset_swap_spread']:.4%}")
    print(f"Opportunité d'arbitrage: {'Oui' if result['arbitrage_opportunity'] else 'Non'}")
    if result['arbitrage_opportunity']:
        print(f"Stratégie: {result['strategy']}")
        print(f"Profit estimé (après coûts): {result['estimated_profit_after_costs']:.4f}")
    
    # ==============================
    # 4. Analyse de sensibilité
    # ==============================
    print("\n\n4. ANALYSE DE SENSIBILITÉ AU SPREAD DE CRÉDIT")
    print("-" * 40)
    
    # Paramètres de l'obligation
    bond_maturity = 7.0    # Maturité de 7 ans
    bond_coupon = 0.045    # Coupon de 4.5%
    
    # Spreads de crédit à analyser
    credit_spreads = np.linspace(0.0, 0.02, 20)  # 0% à 2%
    
    # Résultats pour différents spreads
    bond_prices = []
    bond_ytms = []
    swap_rates = []
    asset_swap_spreads = []
    
    for spread in credit_spreads:
        # Pricer avec spread de crédit
        bond_pricer_with_spread = BondPricer(vasicek_model, credit_spread=spread)
        
        # Prix de l'obligation avec spread
        bond_price = bond_pricer_with_spread.price_fixed_coupon_bond(
            maturity=bond_maturity,
            coupon_rate=bond_coupon
        )
        
        # YTM de l'obligation
        bond_ytm = bond_pricer.calculate_yield_to_maturity(
            bond_price=bond_price,
            maturity=bond_maturity,
            coupon_rate=bond_coupon
        )
        
        # Taux de swap
        swap_rate = swap_pricer.par_rate(0, bond_maturity)
        
        # Spread d'asset swap
        adjusted_coupon = bond_coupon * 100 / bond_price
        asset_swap_spread = adjusted_coupon - swap_rate
        
        # Stockage des résultats
        bond_prices.append(bond_price)
        bond_ytms.append(bond_ytm)
        swap_rates.append(swap_rate)
        asset_swap_spreads.append(asset_swap_spread)
    
    # Graphique des résultats
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Prix et YTM
    ax1.plot(credit_spreads * 100, bond_prices, label='Prix de l\'obligation')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(credit_spreads * 100, np.array(bond_ytms) * 100, 'r--', label='YTM')
    ax1_twin.plot(credit_spreads * 100, np.array(swap_rates) * 100, 'g--', label='Taux de swap')
    
    ax1.set_ylabel('Prix de l\'obligation')
    ax1_twin.set_ylabel('Taux (%)')
    ax1.set_title('Impact du spread de crédit sur le prix de l\'obligation et les taux')
    
    # Légende combinée
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Spread d'asset swap
    ax2.plot(credit_spreads * 100, np.array(asset_swap_spreads) * 100, 'b-', label='Spread d\'asset swap')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Spread de crédit (%)')
    ax2.set_ylabel('Spread d\'asset swap (%)')
    ax2.set_title('Impact du spread de crédit sur le spread d\'asset swap')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('credit_spread_analysis.png')
    plt.close()
    
    print("Analyse de sensibilité au spread de crédit terminée.")
    print("Un graphique a été sauvegardé dans 'credit_spread_analysis.png'.")

if __name__ == "__main__":
    main()