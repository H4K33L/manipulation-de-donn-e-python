import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("titanic/train.csv")

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist([df[df['Survived'] == 0]['Age'].dropna(), df[df['Survived'] == 1]['Age'].dropna()], 
        bins=20, label=['Décédés', 'Survivants'], color=['#FF6666', '#66CC66'], alpha=0.7)

ax.set_xlabel('Âge')
ax.set_ylabel('Nombre de passagers')
ax.set_title("Nombre de survivants et de personnes décédées en fonction de l'âge")
ax.legend()

plt.tight_layout()
plt.show()
