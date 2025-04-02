import numpy as np
import matplotlib.pyplot as plt

# Import des modules nécessaires du projet
from models.interest_rate import Vasicek
from models.derivatives import IRSwapPricer, CapFloorPricer, SwaptionPricer
from instruments.caps_floors import Cap, Floor, Collar
from instruments.swaptions import Swaption

def main():
    # Paramètres du modèle de Vasicek
    r0 = 0.03       # Taux initial
    kappa = 0.5     # Vitesse de retour à la moyenne
    theta = 0.05    # Niveau moyen à long terme
    sigma = 0.01    # Volatilité
    
    # Création du modèle de taux
    vasicek_model = Vasicek(r0, kappa, theta, sigma)
    
    # Création des pricers
    swap_pricer = IRSwapPricer(vasicek_model)
    capfloor_pricer = CapFloorPricer(vasicek_model)
    swaption_pricer = SwaptionPricer(vasicek_model, swap_pricer)
    
    print("=" * 50)
    print("PRICING ET ANALYSE DES OPTIONS SUR TAUX")
    print("=" * 50)
    
    # ==============================
    # Pricing de Cap et Floor
    # ==============================
    print("\n1. PRICING DE CAP ET FLOOR")
    print("-" * 30)
    
    # Paramètres communs
    start_date = 0.0     # Aujourd'hui
    maturity_date = 5.0  # 5 ans
    payment_frequency = 0.5  # Paiements semestriels
    notional = 1000000   # 1 million
    
    # Création d'un Cap
    cap_strike = 0.045   # 4.5%
    cap = Cap(
        start_date=start_date,
        maturity_date=maturity_date,
        strike=cap_strike,
        payment_frequency=payment_frequency,
        notional=notional
    )
    
    # Création d'un Floor
    floor_strike = 0.025  # 2.5%
    floor = Floor(
        start_date=start_date,
        maturity_date=maturity_date,
        strike=floor_strike,
        payment_frequency=payment_frequency,
        notional=notional
    )
    
    # Création d'un Collar
    collar = Collar(
        start_date=start_date,
        maturity_date=maturity_date,
        cap_strike=cap_strike,
        floor_strike=floor_strike,
        payment_frequency=payment_frequency,
        notional=notional
    )
    
    # Analyse de sensibilité à la volatilité
    volatilities = np.linspace(0.005, 0.03, 20)  # Volatilités de 0.5% à 3%
    cap_prices = []
    floor_prices = []
    collar_prices = []
    
    for vol in volatilities:
        cap_price = cap.price(capfloor_pricer, volatility=vol)
        floor_price = floor.price(capfloor_pricer, volatility=vol)
        collar_price = collar.price(capfloor_pricer, volatility=vol)
        
        cap_prices.append(cap_price)
        floor_prices.append(floor_price)
        collar_prices.append(collar_price)
    
    # Affichage des prix
    vol_base = 0.015  # 1.5% volatilité de base
    cap_price_base = cap.price(capfloor_pricer, volatility=vol_base)
    floor_price_base = floor.price(capfloor_pricer, volatility=vol_base)
    collar_price_base = collar.price(capfloor_pricer, volatility=vol_base)
    
    print(f"Prix du Cap (Strike = {cap_strike:.2%}, Vol = {vol_base:.2%}): {cap_price_base:,.2f}")
    print(f"Prix du Floor (Strike = {floor_strike:.2%}, Vol = {vol_base:.2%}): {floor_price_base:,.2f}")
    print(f"Prix du Collar (Cap = {cap_strike:.2%}, Floor = {floor_strike:.2%}, Vol = {vol_base:.2%}): {collar_price_base:,.2f}")
    
    # Graphique de sensibilité à la volatilité
    plt.figure(figsize=(10, 6))
    plt.plot(volatilities * 100, cap_prices, label='Cap')
    plt.plot(volatilities * 100, floor_prices, label='Floor')
    plt.plot(volatilities * 100, collar_prices, label='Collar')
    plt.title('Sensibilité des prix à la volatilité')
    plt.xlabel('Volatilité (%)')
    plt.ylabel('Prix')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('capfloor_vol_sensitivity.png')
    plt.close()
    
    # Analyse de sensibilité au taux sous-jacent
    rates = np.linspace(0.01, 0.07, 30)  # Taux de 1% à 7%
    cap_prices = []
    floor_prices = []
    collar_prices = []
    
    for rate in rates:
        # Mise à jour du taux dans le modèle
        vasicek_model.r0 = rate
        
        cap_price = cap.price(capfloor_pricer, volatility=vol_base)
        floor_price = floor.price(capfloor_pricer, volatility=vol_base)
        collar_price = collar.price(capfloor_pricer, volatility=vol_base)
        
        cap_prices.append(cap_price)
        floor_prices.append(floor_price)
        collar_prices.append(collar_price)
    
    # Graphique de sensibilité au taux sous-jacent
    plt.figure(figsize=(10, 6))
    plt.plot(rates * 100, cap_prices, label='Cap')
    plt.plot(rates * 100, floor_prices, label='Floor')
    plt.plot(rates * 100, collar_prices, label='Collar')
    plt.axvline(x=cap_strike * 100, color='r', linestyle='--', alpha=0.5, label=f'Cap Strike ({cap_strike:.2%})')
    plt.axvline(x=floor_strike * 100, color='g', linestyle='--', alpha=0.5, label=f'Floor Strike ({floor_strike:.2%})')
    plt.title('Sensibilité des prix au taux sous-jacent')
    plt.xlabel('Taux sous-jacent (%)')
    plt.ylabel('Prix')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('capfloor_rate_sensitivity.png')
    plt.close()
    
    # Restauration du taux initial
    vasicek_model.r0 = r0
    
    # ==============================
    # Pricing de Swaption
    # ==============================
    print("\n2. PRICING DE SWAPTION")
    print("-" * 30)
    
    # Paramètres de la swaption
    expiry_date = 2.0  # Expiration dans 2 ans
    underlying_swap_maturity = 5.0  # Swap sous-jacent de 5 ans
    
    # Calcul du taux de swap forward pour le swap sous-jacent
    forward_swap_rate = swap_pricer.par_rate(expiry_date, expiry_date + underlying_swap_maturity, payment_frequency)
    print(f"Taux de swap forward à {expiry_date} ans: {forward_swap_rate:.4%}")
    
    # Création d'une swaption avec un strike proche du forward rate
    swaption_strike = round(forward_swap_rate * 100) / 100  # Arrondi pour simplifier
    
    payer_swaption = Swaption(
        expiry_date=expiry_date,
        underlying_swap_maturity=underlying_swap_maturity,
        strike_rate=swaption_strike,
        is_payer=True,  # Swaption payeuse
        payment_frequency=payment_frequency,
        notional=notional
    )
    
    receiver_swaption = Swaption(
        expiry_date=expiry_date,
        underlying_swap_maturity=underlying_swap_maturity,
        strike_rate=swaption_strike,
        is_payer=False,  # Swaption receveuse
        payment_frequency=payment_frequency,
        notional=notional
    )
    
    # Analyse de sensibilité à la volatilité
    volatilities = np.linspace(0.005, 0.03, 20)  # Volatilités de 0.5% à 3%
    payer_prices = []
    receiver_prices = []
    
    for vol in volatilities:
        payer_price = payer_swaption.price(swaption_pricer, volatility=vol)
        receiver_price = receiver_swaption.price(swaption_pricer, volatility=vol)
        
        payer_prices.append(payer_price)
        receiver_prices.append(receiver_price)
    
    # Affichage des prix
    payer_price_base = payer_swaption.price(swaption_pricer, volatility=vol_base)
    receiver_price_base = receiver_swaption.price(swaption_pricer, volatility=vol_base)
    
    print(f"Prix de la Swaption Payeuse (Strike = {swaption_strike:.2%}, Vol = {vol_base:.2%}): {payer_price_base:,.2f}")
    print(f"Prix de la Swaption Receveuse (Strike = {swaption_strike:.2%}, Vol = {vol_base:.2%}): {receiver_price_base:,.2f}")
    
    # Graphique de sensibilité à la volatilité
    plt.figure(figsize=(10, 6))
    plt.plot(volatilities * 100, payer_prices, label='Swaption Payeuse')
    plt.plot(volatilities * 100, receiver_prices, label='Swaption Receveuse')
    plt.title('Sensibilité des prix de swaption à la volatilité')
    plt.xlabel('Volatilité (%)')
    plt.ylabel('Prix')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('swaption_vol_sensitivity.png')
    plt.close()
    
    # Analyse de sensibilité au taux sous-jacent
    rates = np.linspace(0.01, 0.07, 30)  # Taux de 1% à 7%
    payer_prices = []
    receiver_prices = []
    
    for rate in rates:
        # Mise à jour du taux dans le modèle
        vasicek_model.r0 = rate
        
        payer_price = payer_swaption.price(swaption_pricer, volatility=vol_base)
        receiver_price = receiver_swaption.price(swaption_pricer, volatility=vol_base)
        
        payer_prices.append(payer_price)
        receiver_prices.append(receiver_price)
    
    # Graphique de sensibilité au taux sous-jacent
    plt.figure(figsize=(10, 6))
    plt.plot(rates * 100, payer_prices, label='Swaption Payeuse')
    plt.plot(rates * 100, receiver_prices, label='Swaption Receveuse')
    plt.axvline(x=swaption_strike * 100, color='r', linestyle='--', alpha=0.5, label=f'Strike ({swaption_strike:.2%})')
    plt.title('Sensibilité des prix de swaption au taux sous-jacent')
    plt.xlabel('Taux sous-jacent (%)')
    plt.ylabel('Prix')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('swaption_rate_sensitivity.png')
    plt.close()
    
    print("\nAnalyse des options terminée. Les graphiques ont été sauvegardés.")

if __name__ == "__main__":
    main()