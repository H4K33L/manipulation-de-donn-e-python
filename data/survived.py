import pandas as pd

df = pd.read_csv("../titanic/train.csv")

total_passengers = df.shape[0]

sum_died = (df["Survived"] == 0).sum()
sum_alive = (df["Survived"] == 1).sum()

av_total = 1
av_died = sum_died / total_passengers
av_alive = sum_alive / total_passengers

print(f"Nombre total de passagers : {total_passengers}")
print(f"Nombre de personnes décédées : {sum_died}")
print(f"Nombre de survivants : {sum_alive}")
print(f"Moyenne (proportion) de personnes décédées : {av_died:.2%}")
print(f"Moyenne (proportion) de personnes en vie : {av_alive:.2%}")

survied = pd.DataFrame({
    "category": ["total_passengers", "died", "survived"],
    "sum": [total_passengers, sum_died, sum_alive],
    "average": [av_total, round(av_died, 2), round(av_alive, 2)]
})

survied.to_csv("output/survived.csv", index=False)

print("Tout est dans 'survived.csv' !")
