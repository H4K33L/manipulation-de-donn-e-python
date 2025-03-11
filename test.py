import sys
import pandas as pd 

print("Version Python :", sys.version)
print("Version pandas :", pd.__version__)
print("Installation OK, bon démarrage !")

df = pd.read_csv("titanic/train.csv")

with open("survival_stats.txt", "w") as f:
    for pclass in sorted(df["Pclass"].unique()):
        globals()[f"dead_pclass_{pclass}"] = df[(df["Survived"] == 0) & (df["Pclass"] == pclass)]
        globals()[f"alive_pclass_{pclass}"] = df[(df["Survived"] == 1) & (df["Pclass"] == pclass)]
        
        sum_dead = len(globals()[f"dead_pclass_{pclass}"])
        sum_alive = len(globals()[f"alive_pclass_{pclass}"])
        total = sum_dead + sum_alive
        
        stats = (f"\nClasse {pclass} :\n"
                 f"Number of dead people : {sum_dead} ({(sum_dead / total) * 100:.2f}%)\n"
                 f"Number of survived people : {sum_alive} ({(sum_alive / total) * 100:.2f}%)\n")
        print(stats)
        f.write(stats)