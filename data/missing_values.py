import pandas as pd
import numpy as np

def diagnostic_valeurs_manquantes(df, output_file):
    missing_by_column = df.isnull().sum()
    missing_by_row = df.isnull().sum(axis=1)
    
    total_cells = np.prod(df.shape)
    total_missing = missing_by_column.sum()
    missing_percentage = (total_missing / total_cells) * 100
    
    column_threshold = 0.5 
    row_threshold = 0.5
    
    columns_to_drop = missing_by_column[missing_by_column / df.shape[0] > column_threshold].index
    rows_to_drop = missing_by_row[missing_by_row / df.shape[1] > row_threshold].index
    
    columns_to_impute = missing_by_column[missing_by_column / df.shape[0] <= column_threshold].index
    
    with open(output_file, 'w') as f:
        f.write("Diagnostic des valeurs manquantes\n")
        f.write("="*30 + "\n")
        f.write(f"\nNombre total de valeurs manquantes : {total_missing} sur {total_cells} cellules ({missing_percentage:.2f}%)\n")
        f.write("="*30 + "\n")
        
        f.write("\nNombre de valeurs manquantes par colonne :\n")
        f.write("Colonne | Valeurs manquantes\n")
        f.write("-" * 30 + "\n")
        for col, missing in missing_by_column.items():
            f.write(f"{col:<10} | {missing}\n")
        
        f.write("="*30 + "\n")
        
        f.write("\nNombre de valeurs manquantes par ligne :\n")
        f.write("Ligne | Valeurs manquantes\n")
        f.write("-" * 30 + "\n")
        for idx, missing in missing_by_row.items():
            f.write(f"{idx:<6} | {missing}\n")
        
        f.write("="*30 + "\n")
        
        f.write("\nColonnes avec plus de 50% de valeurs manquantes, a supprimer :\n")
        if len(columns_to_drop) > 0:
            for col in columns_to_drop:
                f.write(f"{col}\n")
        else:
            f.write("Aucune colonne à supprimer.\n")
        
        f.write("="*30 + "\n")
        
        f.write("\nLignes avec plus de 50% de valeurs manquantes, a supprimer :\n")
        if len(rows_to_drop) > 0:
            for row in rows_to_drop:
                f.write(f"{row}\n")
        else:
            f.write("Aucune ligne à supprimer.\n")
        
        f.write("="*30 + "\n")
        
        f.write("\nColonnes à imputer (moins de 50% de valeurs manquantes) :\n")
        if len(columns_to_impute) > 0:
            for col in columns_to_impute:
                f.write(f"{col}\n")
        else:
            f.write("Aucune colonne à imputer.\n")
        
    print(f"Diagnostic ecrit dans le fichier : {output_file}")

if __name__ == "__main__":
    input_file = "../titanic/train.csv"
    df = pd.read_csv(input_file)
    
    output_file = "output/diagnostic_valeurs_manquantes.txt"
    
    diagnostic_valeurs_manquantes(df, output_file)
