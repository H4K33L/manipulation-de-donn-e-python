import pandas as pd

df = pd.read_csv("titanic/test.csv")

print(len(df["PassengerId"]))
for pclass in sorted(df["Pclass"].unique()):
        
        number_classes = len(df[df["Pclass"] == pclass])
        
        stats = (f"\nClasse {pclass} :\n"
                 f"Number of  people : {number_classes})\n")
        print(stats)