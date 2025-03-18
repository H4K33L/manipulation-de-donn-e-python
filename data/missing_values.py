import pandas as pd
import numpy as np

def diagnostic_valeurs_manquantes(df, output_file):
    # Calculer le nombre de valeurs manquantes par ligne et par colonne
    missing_by_column = df.isnull().sum()
    missing_by_row = df.isnull().sum(axis=1)
    
    # Analyser les pourcentages de valeurs manquantes
    total_cells = np.prod(df.shape)
    total_missing = missing_by_column.sum()
    missing_percentage = (total_missing / total_cells) * 100
    
    # Seuils pour décider de supprimer des lignes/colonnes ou imputer
    column_threshold = 0.5  # Seuil pour supprimer une colonne (50% de valeurs manquantes)
    row_threshold = 0.5  # Seuil pour supprimer une ligne (50% de valeurs manquantes)
    
    # Colonnes avec plus de 50% de valeurs manquantes
    columns_to_drop = missing_by_column[missing_by_column / df.shape[0] > column_threshold].index
    rows_to_drop = missing_by_row[missing_by_row / df.shape[1] > row_threshold].index
    
    # Colonnes à imputer (moins de 50% de valeurs manquantes)
    columns_to_impute = missing_by_column[missing_by_column / df.shape[0] <= column_threshold].index
    
    # Ecriture dans le fichier texte
    with open(output_file, 'w') as f:
        # Écrire les informations de base
        f.write("Diagnostic des valeurs manquantes\n")
        f.write("="*30 + "\n")
        f.write(f"\nNombre total de valeurs manquantes : {total_missing} sur {total_cells} cellules ({missing_percentage:.2f}%)\n")
        f.write("="*30 + "\n")
        
        # Tableaux des valeurs manquantes par colonne
        f.write("\nNombre de valeurs manquantes par colonne :\n")
        f.write("Colonne | Valeurs manquantes\n")
        f.write("-" * 30 + "\n")
        for col, missing in missing_by_column.items():
            f.write(f"{col:<10} | {missing}\n")
        
        f.write("="*30 + "\n")
        
        # Tableaux des valeurs manquantes par ligne
        f.write("\nNombre de valeurs manquantes par ligne :\n")
        f.write("Ligne | Valeurs manquantes\n")
        f.write("-" * 30 + "\n")
        for idx, missing in missing_by_row.items():
            f.write(f"{idx:<6} | {missing}\n")
        
        f.write("="*30 + "\n")
        
        # Colonnes à supprimer (plus de 50% de valeurs manquantes)
        f.write("\nColonnes avec plus de 50% de valeurs manquantes, à supprimer :\n")
        if len(columns_to_drop) > 0:
            for col in columns_to_drop:
                f.write(f"{col}\n")
        else:
            f.write("Aucune colonne à supprimer.\n")
        
        f.write("="*30 + "\n")
        
        # Lignes à supprimer (plus de 50% de valeurs manquantes)
        f.write("\nLignes avec plus de 50% de valeurs manquantes, à supprimer :\n")
        if len(rows_to_drop) > 0:
            for row in rows_to_drop:
                f.write(f"{row}\n")
        else:
            f.write("Aucune ligne à supprimer.\n")
        
        f.write("="*30 + "\n")
        
        # Colonnes à imputer (moins de 50% de valeurs manquantes)
        f.write("\nColonnes à imputer (moins de 50% de valeurs manquantes) :\n")
        if len(columns_to_impute) > 0:
            for col in columns_to_impute:
                f.write(f"{col}\n")
        else:
            f.write("Aucune colonne à imputer.\n")
        
    print(f"Diagnostic écrit dans le fichier : {output_file}")

# Exemple d'utilisation
if __name__ == "__main__":
    # Charger un fichier CSV
    input_file = "../titanic/test.csv"  # Remplacer par le chemin du fichier CSV
    df = pd.read_csv(input_file)
    
    # Nom du fichier de sortie
    output_file = "output/diagnostic_valeurs_manquantes.txt"
    
    # Appeler la fonction de diagnostic et écrire dans le fichier
    diagnostic_valeurs_manquantes(df, output_file)
