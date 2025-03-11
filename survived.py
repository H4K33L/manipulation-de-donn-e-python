import pandas as pd

df = pd.read_csv("titanic/train.csv")

sum_died = (df["Survived"] == 0).sum()
sum_alive = (df["Survived"] == 1).sum()

av_died = (df["Survived"] == 0).mean()
av_alive = (df["Survived"] == 1).mean()

print(f"Nombre de personnes décédées : {sum_died}")
print(f"Nombre de survivants : {sum_alive}")
print(f"Moyenne (proportion) de personnes décédées : {av_died:.2%}")
print(f"Moyenne (proportion) de personnes en vie : {av_alive:.2%}")

survied = pd.DataFrame({
    "died_or_survived": ["died", "survived"],
    "sum": [sum_died, sum_alive],
    "average": [round(av_died, 2), round(av_alive, 2)]
})

survied.to_csv("survived.csv", index=False)

print("Les données ont été enregistrées dans 'survived.csv'.")
