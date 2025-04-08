import pandas as pd
from src.preprocessor import TitanicPreprocessor

data = {
    'Age': [22, 38, 26, 35, None, 54, 2, None],
    'Fare': [7.25, 71.2833, 7.925, 8.05, None, 51.8625, 21.075, None],
    'Embarked': ['S', 'C', 'S', 'S', 'Q', 'C', 'S', 'S'],
    'Name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    'Sex': ['male', 'female', 'female', 'male', 'female', 'male', 'female', 'male'],
    'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450', '330877', '17463', '349909'],
    'Cabin': [None, 'C85', None, 'C123', None, 'E46', None, 'B28']
}

df = pd.DataFrame(data)
preprocessor = TitanicPreprocessor()

def test_fit():
    preprocessor.fit(df)
    assert preprocessor.age_median == 31.0, f"Erreur: la médiane de l'âge attendue est 31, mais obtenue {preprocessor.age_median}"
    assert preprocessor.fare_median == 14.4542, f"Erreur: la médiane de la Fare attendue est 14.4542, mais obtenue {preprocessor.fare_median}"
    assert preprocessor.most_frequent_embarkement == 'S', f"Erreur: l'embarquement le plus fréquent attendu est 'S', mais obtenu {preprocessor.most_frequent_embarkement}"

def test_transform():
    preprocessor.fit(df)
    df_transformed = preprocessor.transform(df)
    assert df_transformed['Age'].isna().sum() == 0, "Erreur: des valeurs manquantes existent encore dans la colonne 'Age'"
    assert df_transformed['Fare'].isna().sum() == 0, "Erreur: des valeurs manquantes existent encore dans la colonne 'Fare'"
    assert df_transformed['Embarked'].isna().sum() == 0, "Erreur: des valeurs manquantes existent encore dans la colonne 'Embarked'"
    assert 'Name' not in df_transformed.columns, "Erreur: la colonne 'Name' n'a pas été supprimée"
    assert 'Sex' not in df_transformed.columns, "Erreur: la colonne 'Sex' n'a pas été supprimée"
    assert 'Ticket' not in df_transformed.columns, "Erreur: la colonne 'Ticket' n'a pas été supprimée"
    assert 'Cabin' not in df_transformed.columns, "Erreur: la colonne 'Cabin' n'a pas été supprimée"
    assert 'isMale' in df_transformed.columns, "Erreur: la colonne 'isMale' n'a pas été créée"
    assert df_transformed['isMale'].iloc[0] == 1, "Erreur: la valeur de 'isMale' pour le premier enregistrement devrait être 1 (male)"
    assert df_transformed['isMale'].iloc[1] == 0, "Erreur: la valeur de 'isMale' pour le deuxième enregistrement devrait être 0 (female)"
    assert 'Embarked_C' in df_transformed.columns, "Erreur: la colonne 'Embarked_C' n'est pas présente dans les variables dummy"
    assert 'Embarked_S' in df_transformed.columns, "Erreur: la colonne 'Embarked_S' n'est pas présente dans les variables dummy"

def test_fit_transform():
    df_fit_transformed = preprocessor.fit_transform(df)
    df_transformed = preprocessor.transform(df)
    assert df_fit_transformed.shape == df_transformed.shape, "Erreur: les formes des DataFrames après fit_transform et transform ne correspondent pas"
    assert df_fit_transformed.equals(df_transformed), "Erreur: les DataFrames après fit_transform et transform ne sont pas identiques"

print("Tous les tests de preprocessor ont réussi !")