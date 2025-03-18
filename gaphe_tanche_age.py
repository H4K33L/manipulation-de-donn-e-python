import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('titanic/train.csv')

data = data.dropna(subset=['Age'])
age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
age_labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
data['AgeGroup'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, right=False)

age_group_counts = data['AgeGroup'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
age_group_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Nombre de passagers par tranche d\'âge')
plt.xlabel('Tranche d\'âge')
plt.ylabel('Nombre de passagers')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()