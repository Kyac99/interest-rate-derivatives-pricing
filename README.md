# Projet de Pricing et Gestion des Produits Dérivés de Taux

Ce projet implémente des modèles de pricing pour des instruments dérivés de taux d'intérêt et fournit des outils pour la gestion des risques associés.

## Objectifs

- Pricing précis des swaps de taux (IRS), caps, floors et swaptions
- Simulation du hedging dynamique des expositions aux taux d'intérêt
- Analyse des opportunités d'arbitrage entre obligations et dérivés de taux

## Structure du Projet

```
interest-rate-derivatives-pricing/
├── models/                  # Modèles de taux d'intérêt et pricing
│   ├── interest_rate/       # Modèles de taux (Vasicek, CIR, etc.)
│   ├── derivatives/         # Pricing des dérivés
│   └── volatility/          # Modèles de volatilité
├── instruments/             # Classes pour les instruments financiers
│   ├── swaps.py             # Swaps de taux (IRS)
│   ├── caps_floors.py       # Caps et floors
│   └── swaptions.py         # Swaptions
├── hedging/                 # Stratégies de couverture dynamique
│   ├── delta_hedge.py       # Delta hedging
│   ├── simulation.py        # Simulation de trajectoires
│   └── optimization.py      # Optimisation des stratégies
├── arbitrage/               # Analyse des opportunités d'arbitrage
│   ├── bond_pricing.py      # Pricing des obligations
│   └── opportunities.py     # Détection des opportunités
├── utils/                   # Utilitaires
│   ├── curve_fitting.py     # Ajustement des courbes
│   ├── market_data.py       # Gestion des données de marché
│   └── visualization.py     # Visualisation des résultats
├── tests/                   # Tests unitaires et d'intégration
├── examples/                # Exemples d'utilisation
└── notebooks/               # Notebooks Jupyter pour la démonstration
```

## Méthodologie

### Modèles de Taux d'Intérêt
- Modèle de Vasicek
- Modèle CIR (Cox-Ingersoll-Ross)
- Modèle Black-Karasinski

### Méthodes de Pricing
- Simulation Monte Carlo
- Arbres binomiaux
- Méthodes de différences finies
- Formules analytiques (quand disponibles)

### Techniques de Hedging
- Delta-Hedging
- Gamma-Hedging
- Optimisation de portefeuille

## Installation

```bash
git clone https://github.com/Kyac99/interest-rate-derivatives-pricing.git
cd interest-rate-derivatives-pricing
pip install -r requirements.txt
```

## Utilisation

Des exemples d'utilisation sont disponibles dans le dossier `examples/`.

## Licence

MIT
