import sys
import pandas as pd

print("Python Version:", sys.version)
print("Pandas Version:", pd.__version__)
print("Installation OK, let's start!")

df = pd.read_csv("titanic/train.csv")

df["Has_Cabin"] = df["Cabin"].notna()

with open("survival_stats.txt", "w") as f:
    total_passengers = len(df)
    total_males = len(df[df["Sex"] == "male"])
    total_females = len(df[df["Sex"] == "female"])

    male_percentage = (total_males / total_passengers) * 100 if total_passengers > 0 else 0
    female_percentage = (total_females / total_passengers) * 100 if total_passengers > 0 else 0

    gender_stats = (f"\nGender distribution in the dataset:\n"
                    f"Number of men: {total_males} ({male_percentage:.2f}%)\n"
                    f"Number of women: {total_females} ({female_percentage:.2f}%)\n")
    
    print(gender_stats)
    f.write(gender_stats)

    for pclass in sorted(df["Pclass"].unique()):
        dead_pclass = df[(df["Survived"] == 0) & (df["Pclass"] == pclass)]
        alive_pclass = df[(df["Survived"] == 1) & (df["Pclass"] == pclass)]
        
        sum_dead = len(dead_pclass)
        sum_alive = len(alive_pclass)
        total_pclass = sum_dead + sum_alive if (sum_dead + sum_alive) > 0 else 1  # Avoid division by zero

        class_subset = df[df["Pclass"] == pclass]
        
        has_cabin_count = len(class_subset[class_subset["Has_Cabin"] == True])
        no_cabin_count = len(class_subset[class_subset["Has_Cabin"] == False])
        total_cabin = has_cabin_count + no_cabin_count if (has_cabin_count + no_cabin_count) > 0 else 1  

        has_cabin_percentage = (has_cabin_count / total_cabin) * 100
        no_cabin_percentage = (no_cabin_count / total_cabin) * 100

        males_in_class = len(class_subset[class_subset["Sex"] == "male"])
        females_in_class = len(class_subset[class_subset["Sex"] == "female"])
        total_sex = males_in_class + females_in_class if (males_in_class + females_in_class) > 0 else 1  

        male_percentage = (males_in_class / total_sex) * 100
        female_percentage = (females_in_class / total_sex) * 100

        dead_males_in_class = len(dead_pclass[dead_pclass["Sex"] == "male"])
        dead_females_in_class = len(dead_pclass[dead_pclass["Sex"] == "female"])

        dead_male_percentage = (dead_males_in_class / sum_dead) * 100 if sum_dead > 0 else 0
        dead_female_percentage = (dead_females_in_class / sum_dead) * 100 if sum_dead > 0 else 0

        survived_males_in_class = len(alive_pclass[alive_pclass["Sex"] == "male"])
        survived_females_in_class = len(alive_pclass[alive_pclass["Sex"] == "female"])

        survived_male_percentage = (survived_males_in_class / sum_alive) * 100 if sum_alive > 0 else 0
        survived_female_percentage = (survived_females_in_class / sum_alive) * 100 if sum_alive > 0 else 0

        stats = (f"\nClass {pclass}:\n"
                 f"Number of deaths: {sum_dead} ({(sum_dead / total_pclass) * 100:.2f}%)\n"
                 f"Number of survivors: {sum_alive} ({(sum_alive / total_pclass) * 100:.2f}%)\n"
                 f"Number of people with a cabin: {has_cabin_count} ({has_cabin_percentage:.2f}%)\n"
                 f"Number of people without a cabin: {no_cabin_count} ({no_cabin_percentage:.2f}%)\n"
                 f"Number of men: {males_in_class} ({male_percentage:.2f}%)\n"
                 f"Number of women: {females_in_class} ({female_percentage:.2f}%)\n"
                 f"Number of dead men: {dead_males_in_class} ({dead_male_percentage:.2f}%)\n"
                 f"Number of dead women: {dead_females_in_class} ({dead_female_percentage:.2f}%)\n"
                 f"Number of surviving men: {survived_males_in_class} ({survived_male_percentage:.2f}%)\n"
                 f"Number of surviving women: {survived_females_in_class} ({survived_female_percentage:.2f}%)\n")

        print(stats)
        f.write(stats)

    for has_cabin in [True, False]:
        label = "with cabin" if has_cabin else "without cabin"
        dead_cabin = df[(df["Survived"] == 0) & (df["Has_Cabin"] == has_cabin)]
        alive_cabin = df[(df["Survived"] == 1) & (df["Has_Cabin"] == has_cabin)]
        
        sum_dead = len(dead_cabin)
        sum_alive = len(alive_cabin)
        total = sum_dead + sum_alive if (sum_dead + sum_alive) > 0 else 1  

        dead_percentage = (sum_dead / total) * 100
        alive_percentage = (sum_alive / total) * 100

        stats = (f"\nPassengers {label}:\n"
                 f"Number of deaths: {sum_dead} ({dead_percentage:.2f}%)\n"
                 f"Number of survivors: {sum_alive} ({alive_percentage:.2f}%)\n")

        print(stats)
        f.write(stats)
    for gender in ["male", "female"]:
        label = f"{gender}s"
        
        dead_gender = df[(df["Survived"] == 0) & (df["Sex"] == gender)]
        alive_gender = df[(df["Survived"] == 1) & (df["Sex"] == gender)]
        
        sum_dead = len(dead_gender)
        sum_alive = len(alive_gender)
        total = sum_dead + sum_alive if (sum_dead + sum_alive) > 0 else 1  # Avoid division by zero
        
        dead_percentage = (sum_dead / total) * 100
        alive_percentage = (sum_alive / total) * 100

        stats = (f"\n{label.capitalize()} passengers:\n"
                f"Number of deaths: {sum_dead} ({dead_percentage:.2f}%)\n"
                f"Number of survivors: {sum_alive} ({alive_percentage:.2f}%)\n")
        
        print(stats)
        f.write(stats)
