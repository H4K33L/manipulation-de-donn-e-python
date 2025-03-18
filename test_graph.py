import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("titanic/train.csv")

total_passengers = df.shape[0]
sum_died = (df["Survived"] == 0).sum()
sum_alive = (df["Survived"] == 1).sum()

av_total = 1
av_died = sum_died / total_passengers
av_alive = sum_alive / total_passengers

num_men = (df["Sex"] == "male").sum()
num_women = (df["Sex"] == "female").sum()
perc_men = num_men / total_passengers * 100
perc_women = num_women / total_passengers * 100

num_with_cabin = df["Cabin"].notna().sum()
num_without_cabin = df["Cabin"].isna().sum()
perc_with_cabin = num_with_cabin / total_passengers * 100
perc_without_cabin = num_without_cabin / total_passengers * 100

avg_age = df["Age"].mean()

plt.figure(figsize=(6, 6))
plt.bar(["Décédés", "Survivants"], [sum_died, sum_alive], color=['red', 'green'])
plt.title("Répartition des survivants et des victimes")
plt.ylabel("Nombre de personnes")
plt.savefig("survivants_victimes.png")
plt.close()

plt.figure(figsize=(6, 6))
plt.pie([perc_men, perc_women], labels=["Hommes", "Femmes"], autopct='%1.1f%%', colors=['blue', 'pink'])
plt.title("Répartition des sexes à bord")
plt.savefig("repartition_sexes.png")
plt.close()

plt.figure(figsize=(6, 6))
plt.pie([perc_with_cabin, perc_without_cabin], labels=["Avec cabine", "Sans cabine"], autopct='%1.1f%%', colors=['purple', 'gray'])
plt.title("Possession d'une cabine")
plt.savefig("graphique_cabine.png")
plt.close()

plt.figure(figsize=(6, 6))
plt.hist(df["Age"].dropna(), bins=20, color='orange', edgecolor='black')
plt.title("Graphique de l'âge des passagers")
plt.xlabel("Âge")
plt.ylabel("Nombre de passagers")
plt.savefig("graphique_age.png")
plt.close()
