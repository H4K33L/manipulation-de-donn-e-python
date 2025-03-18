import pandas as pd
import matplotlib.pyplot as plt

# Charger les données
data = pd.read_csv('titanic/train.csv')

# Supprimer les lignes avec des valeurs d'âge manquantes
data = data.dropna(subset=['Age'])

# Créer des tranches d'âge
age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
age_labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
data['AgeGroup'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, right=False)

# Calculer le taux de survie par tranche d'âge
survival_rates = data.groupby('AgeGroup')['Survived'].mean()

# Créer le graphe
plt.figure(figsize=(10, 6))
survival_rates.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Taux de survie par tranche d\'âge')
plt.xlabel('Tranche d\'âge')
plt.ylabel('Taux de survie')
plt.ylim(0, 1)  # Fixer l'échelle de 0 à 1 pour le taux de survie
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Ajouter les valeurs sur les barres
for i, v in enumerate(survival_rates):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()