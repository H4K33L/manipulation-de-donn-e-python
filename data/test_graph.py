import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../titanic/train.csv")

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

num_men_with_cabin = df[(df["Sex"] == "male") & (df["Cabin"].notna())].shape[0]
num_men_without_cabin = df[(df["Sex"] == "male") & (df["Cabin"].isna())].shape[0]
num_women_with_cabin = df[(df["Sex"] == "female") & (df["Cabin"].notna())].shape[0]
num_women_without_cabin = df[(df["Sex"] == "female") & (df["Cabin"].isna())].shape[0]

bins = [0, 12, 18, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['0-12', '13-18', '19-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91+']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

age_grouped = df.groupby(['AgeGroup', 'Survived']).size().unstack(fill_value=0)

labels_gender_cabin = ["Hommes avec cabine", "Hommes sans cabine", "Femmes avec cabine", "Femmes sans cabine"]
sizes_gender_cabin = [num_men_with_cabin, num_men_without_cabin, num_women_with_cabin, num_women_without_cabin]
colors_gender_cabin = ['blue', 'lightblue', 'pink', 'lightpink']

plt.figure(figsize=(8, 8))
plt.pie(sizes_gender_cabin, labels=labels_gender_cabin, autopct='%1.1f%%', colors=colors_gender_cabin)
plt.title("Répartition des sexes avec ou sans cabine")
plt.savefig("output/repartition_sexes_cabine.png")
plt.close()

age_grouped.plot(kind='bar', stacked=False, color=['red', 'green'], figsize=(10, 6))
plt.title("Nombre de survivants et décédés par tranche d'âge")
plt.xlabel("Tranche d'âge")
plt.ylabel("Nombre de passagers")
plt.xticks(rotation=45)
plt.legend(["Décédés", "Survivants"], loc="upper left")
plt.tight_layout()
plt.savefig("output/survivants_victimes_par_age.png")
plt.close()
