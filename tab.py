import sys
import pandas as pd

print("Python Version:", sys.version)
print("Pandas Version:", pd.__version__)
print("Installation OK, let's start!")

df = pd.read_csv("titanic/train.csv")
df["Has_Cabin"] = df["Cabin"].notna()

bins = [0, 12, 17, 60, float('inf')]
labels = ['Children (0-12)', 'Teenagers (13-17)', 'Adults (18-60)', 'Seniors (60+)']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

pivot_table = df.pivot_table(
    values='PassengerId', 
    index=['Pclass', 'Sex', 'AgeGroup', 'Has_Cabin'], 
    columns='Survived', 
    aggfunc='count', 
    fill_value=0
)

pivot_table.columns = ['Died', 'Survived']
pivot_table['Total'] = pivot_table['Died'] + pivot_table['Survived']
pivot_table['Survival Rate (%)'] = (pivot_table['Survived'] / pivot_table['Total']) * 100

with open("survival_stats_table.txt", "w", encoding="utf-8") as f:
    f.write(pivot_table.to_string())

