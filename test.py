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
    df["Has_Cabin"] = df["Cabin"].notna()
    for has_cabin in [True, False]:
        label = "avec cabine" if has_cabin else "sans cabine"
        dead_cabin = df[(df["Survived"] == 0) & (df["Has_Cabin"] == has_cabin)]
        alive_cabin = df[(df["Survived"] == 1) & (df["Has_Cabin"] == has_cabin)]
        
        sum_dead = len(dead_cabin)
        sum_alive = len(alive_cabin)
        total = sum_dead + sum_alive
        
        stats = (f"\nPassagers {label} :\n"
                 f"Number of dead people : {sum_dead} ({(sum_dead / total) * 100:.2f}%)\n"
                 f"Number of survived people : {sum_alive} ({(sum_alive / total) * 100:.2f}%)\n")
        
        print(stats)
        f.write(stats)
    for pclass in sorted(df["Pclass"].unique()):
        class_subset = df[df["Pclass"] == pclass]
        has_cabin_count = len(class_subset[class_subset["Has_Cabin"] == True])
        no_cabin_count = len(class_subset[class_subset["Has_Cabin"] == False])
        total = has_cabin_count + no_cabin_count
        
        has_cabin_percentage = (has_cabin_count / total) * 100 if total > 0 else 0.0
        no_cabin_percentage = (no_cabin_count / total) * 100 if total > 0 else 0.0
        
        stats = (f"\nClasse {pclass} :\n"
                 f"Nombre de personnes avec cabine : {has_cabin_count} ({has_cabin_percentage:.2f}%)\n"
                 f"Nombre de personnes sans cabine : {no_cabin_count} ({no_cabin_percentage:.2f}%)\n")
        
        print(stats)
        f.write(stats)