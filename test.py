import pandas as pd
df = pd.read_csv("titanic/train.csv")

sum_dead = (df["Survived"] == 0).sum()
sum_alive = (df["Survived"] == 1).sum()

print(f"Nombre de personnes décédées : {sum_dead}")
print(f"Nombre de survivants : {sum_alive}")

av_dead = (df["Survived"] == 0).mean()
av_alive = (df["Survived"] == 1).mean()

print(f"Moyenne (proportion) de personnes décédées : {av_dead:.2%}")
print(f"Moyenne (proportion) de personnes en vie : {av_alive:.2%}")